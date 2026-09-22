const std = @import("std");

/// Protocol-neutral identity carried across agent turns.
///
/// A binding is only an identifier.  Divergent-prefix reuse is authorized
/// only when a STRONG identifier also resolves to a frontier that this server
/// previously issued/remembered.  Hints are useful for routing/affinity but
/// can never by themselves justify skipping different prompt tokens.
pub const Strength = enum {
    hint,
    strong,
};

pub const Kind = enum(u8) {
    /// OpenAI Responses: previous_response_id.
    previous_response_id,
    /// OpenAI-style tool/function continuation.
    tool_call_id,
    /// Anthropic Messages tool_result -> tool_use id.
    tool_use_id,
    /// OpenAI prompt routing/cache affinity field.
    prompt_cache_key,
    /// Stable per-session/user routing metadata used by several clients.
    metadata_user_id,
};

pub const Binding = struct {
    kind: Kind,
    digest: u64,
    strength: Strength,

    pub fn authorizesDivergentReuse(self: Binding) bool {
        return self.strength == .strong;
    }
};

fn strengthFor(kind: Kind) Strength {
    return switch (kind) {
        .previous_response_id, .tool_call_id, .tool_use_id => .strong,
        .prompt_cache_key, .metadata_user_id => .hint,
    };
}

/// Domain-separated digest: identical bytes in two protocol fields never alias.
pub fn make(kind: Kind, value: []const u8) ?Binding {
    if (value.len == 0) return null;
    var h = std.hash.Wyhash.init(0x4c495645434f4e54); // "LIVECONT"
    h.update(&[_]u8{@intFromEnum(kind)});
    h.update(value);
    return .{
        .kind = kind,
        .digest = h.final(),
        .strength = strengthFor(kind),
    };
}


/// Opaque pointer to an exact hot-cache frontier.  The continuation layer does
/// not know KV/SSM internals; it only remembers enough identity for the model
/// cache to locate the exact entry later.
pub const FrontierRef = struct {
    /// Process-local model identity (model registry assigns this later).
    model_tag: u64,
    /// Exact hot-cache entry identity. Never derived from a client value.
    entry_tag: u64,
    /// Exact model-state frontier represented by the entry.
    token_len: u32,
    /// Workload/session affinity only; not part of the authorization decision.
    cache_key: u64 = 0,
};

const Record = struct {
    kind: Kind,
    digest: u64,
    frontier: FrontierRef,
    stamp: u64,
};

/// Small process-local registry from protocol proof -> exact model frontier.
///
/// Only strong bindings may enter this table.  A session hint can choose/retain
/// a cache lineage, but can never resolve to a divergent live state.
pub const Registry = struct {
    allocator: std.mem.Allocator,
    cap: usize,
    clock: u64 = 0,
    records: std.ArrayListUnmanaged(Record) = .empty,

    pub fn init(allocator: std.mem.Allocator, cap: usize) Registry {
        return .{ .allocator = allocator, .cap = @max(cap, 1) };
    }

    pub fn deinit(self: *Registry) void {
        self.records.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn bind(self: *Registry, binding: Binding, frontier: FrontierRef) !bool {
        if (!binding.authorizesDivergentReuse()) return false;
        self.clock +%= 1;

        for (self.records.items) |*record| {
            if (record.kind == binding.kind and record.digest == binding.digest) {
                record.frontier = frontier;
                record.stamp = self.clock;
                return true;
            }
        }

        if (self.records.items.len >= self.cap) {
            var oldest_i: usize = 0;
            var oldest_stamp = self.records.items[0].stamp;
            for (self.records.items[1..], 1..) |record, i| {
                if (record.stamp < oldest_stamp) {
                    oldest_i = i;
                    oldest_stamp = record.stamp;
                }
            }
            _ = self.records.swapRemove(oldest_i);
        }

        try self.records.append(self.allocator, .{
            .kind = binding.kind,
            .digest = binding.digest,
            .frontier = frontier,
            .stamp = self.clock,
        });
        return true;
    }

    pub fn resolve(self: *Registry, binding: Binding) ?FrontierRef {
        if (!binding.authorizesDivergentReuse()) return null;
        for (self.records.items) |*record| {
            if (record.kind == binding.kind and record.digest == binding.digest) {
                self.clock +%= 1;
                record.stamp = self.clock;
                return record.frontier;
            }
        }
        return null;
    }
};

pub const Surface = enum {
    chat_completions,
    responses,
    anthropic_messages,
};

pub const MAX_BINDINGS: usize = 16;

pub const Set = struct {
    items: [MAX_BINDINGS]Binding = undefined,
    len: usize = 0,

    pub fn slice(self: *const Set) []const Binding {
        return self.items[0..self.len];
    }

    pub fn append(self: *Set, binding: Binding) void {
        for (self.items[0..self.len]) |existing| {
            if (existing.kind == binding.kind and existing.digest == binding.digest) return;
        }
        if (self.len >= self.items.len) return;
        self.items[self.len] = binding;
        self.len += 1;
    }

    pub fn hasStrong(self: *const Set) bool {
        for (self.items[0..self.len]) |binding| {
            if (binding.authorizesDivergentReuse()) return true;
        }
        return false;
    }
};

fn appendStringField(set: *Set, obj: std.json.ObjectMap, field: []const u8, kind: Kind) void {
    const v = obj.get(field) orelse return;
    if (v != .string) return;
    if (make(kind, v.string)) |binding| set.append(binding);
}

fn collectCommon(set: *Set, root: std.json.ObjectMap) void {
    appendStringField(set, root, "prompt_cache_key", .prompt_cache_key);
    if (root.get("metadata")) |metadata| if (metadata == .object) {
        appendStringField(set, metadata.object, "user_id", .metadata_user_id);
    };
}

fn collectChatTools(set: *Set, root: std.json.ObjectMap) void {
    const messages = root.get("messages") orelse return;
    if (messages != .array) return;
    for (messages.array.items) |message| {
        if (message != .object) continue;
        // Tool result continuation: OpenAI chat-completions.
        appendStringField(set, message.object, "tool_call_id", .tool_call_id);
    }
}

fn collectResponses(set: *Set, root: std.json.ObjectMap) void {
    appendStringField(set, root, "previous_response_id", .previous_response_id);
    const input = root.get("input") orelse return;
    if (input != .array) return;
    for (input.array.items) |item| {
        if (item != .object) continue;
        const type_val = item.object.get("type") orelse continue;
        if (type_val != .string or !std.mem.eql(u8, type_val.string, "function_call_output")) continue;
        appendStringField(set, item.object, "call_id", .tool_call_id);
    }
}

fn collectAnthropicTools(set: *Set, root: std.json.ObjectMap) void {
    const messages = root.get("messages") orelse return;
    if (messages != .array) return;
    for (messages.array.items) |message| {
        if (message != .object) continue;
        const content = message.object.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array.items) |part| {
            if (part != .object) continue;
            const type_val = part.object.get("type") orelse continue;
            if (type_val != .string or !std.mem.eql(u8, type_val.string, "tool_result")) continue;
            appendStringField(set, part.object, "tool_use_id", .tool_use_id);
        }
    }
}

/// Extract protocol bindings without naming or depending on a particular agent
/// framework.  The caller may use hints for cache affinity.  A later registry
/// lookup must confirm strong bindings were previously issued by this server
/// before they can authorize a live-frontier continuation.
pub fn collect(root: std.json.ObjectMap, surface: Surface) Set {
    var set: Set = .{};
    collectCommon(&set, root);
    switch (surface) {
        .chat_completions => collectChatTools(&set, root),
        .responses => collectResponses(&set, root),
        .anthropic_messages => collectAnthropicTools(&set, root),
    }
    return set;
}

test "binding strength keeps session hints from authorizing divergence" {
    const strong = make(.previous_response_id, "resp_123").?;
    const weak = make(.prompt_cache_key, "session-123").?;
    try std.testing.expect(strong.authorizesDivergentReuse());
    try std.testing.expect(!weak.authorizesDivergentReuse());
}

test "binding digest is domain separated by protocol field" {
    const a = make(.tool_call_id, "same-id").?;
    const b = make(.tool_use_id, "same-id").?;
    try std.testing.expect(a.digest != b.digest);
}

test "collect covers OpenAI chat, Responses, and Anthropic tool continuations" {
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
            \\{"prompt_cache_key":"sess","messages":[{"role":"tool","tool_call_id":"call_1","content":"ok"}]}
        , .{});
        defer parsed.deinit();
        const set = collect(parsed.value.object, .chat_completions);
        try std.testing.expectEqual(@as(usize, 2), set.len);
        try std.testing.expect(set.hasStrong());
    }
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
            \\{"previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_2","output":"ok"}]}
        , .{});
        defer parsed.deinit();
        const set = collect(parsed.value.object, .responses);
        try std.testing.expectEqual(@as(usize, 2), set.len);
        try std.testing.expect(set.hasStrong());
    }
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
            \\{"metadata":{"user_id":"claude-session"},"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}
        , .{});
        defer parsed.deinit();
        const set = collect(parsed.value.object, .anthropic_messages);
        try std.testing.expectEqual(@as(usize, 2), set.len);
        try std.testing.expect(set.hasStrong());
    }
}


test "registry binds only server-confirmed strong protocol identities" {
    var reg = Registry.init(std.testing.allocator, 4);
    defer reg.deinit();

    const frontier = FrontierRef{
        .model_tag = 11,
        .entry_tag = 22,
        .token_len = 100_123,
        .cache_key = 33,
    };

    const weak = make(.prompt_cache_key, "sess").?;
    try std.testing.expect(!(try reg.bind(weak, frontier)));
    try std.testing.expect(reg.resolve(weak) == null);

    const strong = make(.tool_call_id, "call_abc").?;
    try std.testing.expect(try reg.bind(strong, frontier));
    const got = reg.resolve(strong).?;
    try std.testing.expectEqual(frontier.entry_tag, got.entry_tag);
    try std.testing.expectEqual(frontier.token_len, got.token_len);
}

test "registry is bounded and refreshes resolved bindings" {
    var reg = Registry.init(std.testing.allocator, 2);
    defer reg.deinit();

    const a = make(.previous_response_id, "resp_a").?;
    const b = make(.previous_response_id, "resp_b").?;
    const c = make(.previous_response_id, "resp_c").?;
    _ = try reg.bind(a, .{ .model_tag = 1, .entry_tag = 1, .token_len = 10 });
    _ = try reg.bind(b, .{ .model_tag = 1, .entry_tag = 2, .token_len = 20 });
    _ = reg.resolve(a); // a becomes MRU
    _ = try reg.bind(c, .{ .model_tag = 1, .entry_tag = 3, .token_len = 30 });

    try std.testing.expect(reg.resolve(a) != null);
    try std.testing.expect(reg.resolve(b) == null);
    try std.testing.expect(reg.resolve(c) != null);
}
