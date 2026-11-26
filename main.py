"""IVARA — Personal AI Memory & Reasoning Assistant.

Command-line entry point for interactive chat, conversation management,
and long-term semantic memory management.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from src.conversations.store import ConversationStore, DEFAULT_CONVERSATIONS_DB
from src.graph.store import SQLiteGraphStore
from src.memory import (
    load_memories,
    get_memory,
    list_memories,
    delete_memory,
    clear_memories,
    get_memory_stats,
    sync_memory_graph,
    DATA_PATH as DEFAULT_MEMORIES_PATH
)
from src.ontology.entity_store import DATA_PATH as DEFAULT_ENTITIES_PATH
from src.retrieval import retrieve
from src.answering import answer_question
from src.chat.engine import ChatEngine
from src.chat.terminal import run_interactive_chat


DEFAULT_GRAPH_DB = Path(__file__).resolve().parent / "data" / "memory_graph.db"


def get_default_engine() -> ChatEngine:
    """Instantiate standard production ChatEngine wired to production stores."""
    conv_store = ConversationStore(DEFAULT_CONVERSATIONS_DB)
    graph_store = SQLiteGraphStore(DEFAULT_GRAPH_DB)
    # Ensure existing memories are synchronized into the graph if needed
    if graph_store.count_nodes() == 0 and len(load_memories(DEFAULT_MEMORIES_PATH)) > 0:
        sync_memory_graph(DEFAULT_MEMORIES_PATH, DEFAULT_ENTITIES_PATH, graph_store)
    return ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=DEFAULT_MEMORIES_PATH,
        entities_path=DEFAULT_ENTITIES_PATH
    )


# ---------------------------------------------------------------------------
# Chat Subcommands
# ---------------------------------------------------------------------------

def run_sandbox_chat(conv_id: Optional[str] = None):
    """Run an interactive chat session in an isolated temporary directory without touching production data."""
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp(prefix="ivara_sandbox_"))
    try:
        temp_data = temp_dir / "data"
        temp_data.mkdir(parents=True, exist_ok=True)

        sandbox_conv_db = temp_data / "conversations.db"
        sandbox_memories = temp_data / "memories.json"
        sandbox_entities = temp_data / "entities.json"
        sandbox_graph_db = temp_data / "memory_graph.db"

        sandbox_memories.write_text("[]", encoding="utf-8")
        sandbox_entities.write_text("{}", encoding="utf-8")

        conv_store = ConversationStore(sandbox_conv_db)
        graph_store = SQLiteGraphStore(sandbox_graph_db)
        engine = ChatEngine(
            conversation_store=conv_store,
            graph_store=graph_store,
            memories_path=sandbox_memories,
            entities_path=sandbox_entities,
        )
        print("=" * 80)
        print(" [IVARA SANDBOX MODE]")
        print(f" Isolated runtime storage: {temp_dir}")
        print(" Production databases and memories will NOT be modified or read.")
        print("=" * 80)
        run_interactive_chat(engine=engine, conversation_id=conv_id)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def cmd_chat(args):
    """Launch interactive conversational session."""
    sandbox = getattr(args, "sandbox", False)
    if sandbox:
        run_sandbox_chat(conv_id=getattr(args, "id", None))
        return
    engine = get_default_engine()
    conv_id = getattr(args, "id", None)
    run_interactive_chat(engine=engine, conversation_id=conv_id)


def cmd_chat_open(args):
    """Open and resume an existing conversation session."""
    engine = get_default_engine()
    conv = engine.conv_store.get_conversation(args.id)
    if not conv:
        print(f"Error: Conversation '{args.id}' not found.")
        sys.exit(1)
    run_interactive_chat(engine=engine, conversation_id=args.id)


def cmd_chats_list(args):
    """List all stored conversation sessions."""
    store = ConversationStore(DEFAULT_CONVERSATIONS_DB)
    convs = store.list_conversations(limit=args.limit, offset=args.offset)
    if not convs:
        print("No conversations found.")
        return

    print(f"\nSaved Conversations ({len(convs)}):")
    print("-" * 75)
    print(f"{'ID':<20} | {'Messages':<8} | {'Updated At':<24} | Title")
    print("-" * 75)
    for c in convs:
        m_count = store.count_messages(c.id)
        print(f"{c.id:<20} | {m_count:<8} | {c.updated_at[:19]:<24} | {c.title}")
    print("-" * 75)


def cmd_chat_delete(args):
    """Delete a conversation, optionally purging originated memories."""
    store = ConversationStore(DEFAULT_CONVERSATIONS_DB)
    conv = store.get_conversation(args.id)
    if not conv:
        print(f"Error: Conversation '{args.id}' does not exist.")
        sys.exit(1)

    deleted = store.delete_conversation(args.id)
    if deleted:
        print(f"Successfully deleted conversation '{args.id}'.")
    else:
        print(f"Failed to delete conversation '{args.id}'.")
        sys.exit(1)

    if args.purge_memories:
        graph_store = SQLiteGraphStore(DEFAULT_GRAPH_DB)
        memos = list_memories(DEFAULT_MEMORIES_PATH)
        purged = 0
        prefix = args.id if args.id.startswith("conv_") else f"conv_{args.id}"
        for m in memos:
            src = str(m.get("source", ""))
            if src.startswith(prefix) or src.startswith(f"conv_{prefix}"):
                delete_memory(m["id"], data_path=DEFAULT_MEMORIES_PATH, graph_store=graph_store)
                purged += 1
        print(f"Purged {purged} long-term memories originated from conversation '{args.id}'.")


def cmd_chats_clear(args):
    """Clear all conversation history."""
    if not args.yes:
        confirm = input("Are you sure you want to delete ALL conversations? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Operation aborted.")
            return

    store = ConversationStore(DEFAULT_CONVERSATIONS_DB)
    cleared = store.clear_conversations()
    print(f"Successfully cleared {cleared} conversation sessions.")


# ---------------------------------------------------------------------------
# Memories Subcommands
# ---------------------------------------------------------------------------

def cmd_memories_list(args):
    """List all long-term semantic memories."""
    memos = list_memories(DEFAULT_MEMORIES_PATH)
    if not memos:
        print("No memories stored.")
        return

    print(f"\nLong-Term Memories ({len(memos)} total):")
    print("-" * 80)
    print(f"{'ID':<6} | {'Date':<10} | {'Type':<12} | Text")
    print("-" * 80)
    for m in memos[-args.limit:]:
        mid = m.get("id", "???")
        ts = str(m.get("timestamp", ""))[:10]
        mtype = str(m.get("type", "fact"))[:12]
        txt = str(m.get("text", "")).replace("\n", " ")
        if len(txt) > 48:
            txt = txt[:45] + "..."
        print(f"{mid:<6} | {ts:<10} | {mtype:<12} | {txt}")
    print("-" * 80)


def cmd_memories_search(args):
    """Search memories using semantic retrieval."""
    results = retrieve(query=args.query, k=args.k, min_score=args.min_score, data_path=DEFAULT_MEMORIES_PATH)
    if not results:
        print(f"No memories matched '{args.query}' (min_score={args.min_score}).")
        return

    print(f"\nSearch Results for: \"{args.query}\" ({len(results)} matches)")
    print("-" * 80)
    for r in results:
        score = r.get("score", 0.0)
        print(f"[{r.get('id')}] Score: {score:.3f} | Date: {r.get('timestamp')}")
        print(f"  Text: {r.get('text')}")
        if r.get("entities"):
            ents = [e.get("canonical_name", e.get("name", "")) for e in r["entities"]]
            print(f"  Entities: {', '.join(filter(None, ents))}")
        print()
    print("-" * 80)


def cmd_memories_show(args):
    """Show detailed metadata and structure of a specific memory."""
    memo = get_memory(args.id, data_path=DEFAULT_MEMORIES_PATH)
    if not memo:
        print(f"Error: Memory '{args.id}' not found.")
        sys.exit(1)

    print(f"\nMemory Details: {memo.get('id')}")
    print("=" * 60)
    print(f"Text:         {memo.get('text')}")
    print(f"Timestamp:    {memo.get('timestamp')}")
    print(f"Source:       {memo.get('source')}")
    print(f"Type:         {memo.get('type')}")
    print(f"Importance:   {memo.get('importance')}")
    print(f"Confidence:   {memo.get('confidence')}")

    entities = memo.get("entities", [])
    print(f"\nEntities ({len(entities)}):")
    for e in entities:
        cname = e.get("canonical_name", e.get("mention", ""))
        types = e.get("types", [])
        print(f"  - {cname} (types: {types})")

    relationships = memo.get("relationships", [])
    print(f"\nRelationships ({len(relationships)}):")
    for rel in relationships:
        src = rel.get("source", "")
        pred = rel.get("relation", "")
        tgt = rel.get("target", "")
        origin = rel.get("origin", "")
        print(f"  - {src} --[{pred}]--> {tgt} ({origin})")

    claims = memo.get("temporal_claims", [])
    if claims:
        print(f"\nTemporal Claims ({len(claims)}):")
        for cl in claims:
            vf = cl.get("valid_from", "none")
            vt = cl.get("valid_to", "present")
            print(f"  - [{cl.get('claim_id')}] {cl.get('subject')} {cl.get('predicate')} {cl.get('object')} (valid: {vf} -> {vt})")
    print("=" * 60)


def cmd_memories_delete(args):
    """Delete a memory from JSON and purge its graph representations."""
    graph_store = SQLiteGraphStore(DEFAULT_GRAPH_DB)
    deleted = delete_memory(args.id, data_path=DEFAULT_MEMORIES_PATH, graph_store=graph_store)
    if deleted:
        print(f"Successfully deleted memory '{args.id}' from memory store and graph database.")
    else:
        print(f"Error: Memory '{args.id}' not found.")
        sys.exit(1)


def cmd_memories_clear(args):
    """Clear all long-term memories and graph representations."""
    if not args.yes:
        confirm = input("Are you sure you want to delete ALL long-term memories and graph structures? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Operation aborted.")
            return

    graph_store = SQLiteGraphStore(DEFAULT_GRAPH_DB)
    cleared = clear_memories(data_path=DEFAULT_MEMORIES_PATH, graph_store=graph_store)
    print(f"Successfully cleared {cleared} long-term memories and wiped graph store.")


def cmd_memories_stats(args):
    """Display statistical overview of memories, entities, and graph store."""
    graph_store = SQLiteGraphStore(DEFAULT_GRAPH_DB)
    if graph_store.count_nodes() == 0 and len(load_memories(DEFAULT_MEMORIES_PATH)) > 0:
        sync_memory_graph(DEFAULT_MEMORIES_PATH, DEFAULT_ENTITIES_PATH, graph_store)
    stats = get_memory_stats(data_path=DEFAULT_MEMORIES_PATH, graph_store=graph_store)
    conv_store = ConversationStore(DEFAULT_CONVERSATIONS_DB)

    print("\nIVARA Memory System Overview:")
    print("=" * 45)
    print(f"  Long-Term Memories:        {stats.get('total_memories', 0)}")
    print(f"  Unique Memory Sources:     {stats.get('unique_sources', 0)}")
    print(f"  Entities in Memories:      {stats.get('entities_in_memories', 0)}")
    print(f"  Relationships in Memories: {stats.get('relationships_in_memories', 0)}")
    print(f"  Temporal Claims:           {stats.get('temporal_claims_in_memories', 0)}")
    print(f"  Graph Nodes (SQLite):      {stats.get('graph_nodes', 0)}")
    print(f"  Graph Edges (SQLite):      {stats.get('graph_edges', 0)}")
    print(f"  Active Chat Sessions:      {conv_store.count_conversations()}")
    print(f"  Total Chat Messages:       {conv_store.count_messages()}")
    print("=" * 45)


def cmd_ask(args):
    """Answer an evidence-grounded personal memory question."""
    engine = get_default_engine()
    use_adaptive = getattr(args, "adaptive", False)
    res = answer_question(
        query=args.query,
        k=args.k,
        min_score=args.min_score,
        data_path=engine.memories_path,
        model=getattr(args, "model", engine.model),
        use_graph=True,
        graph_store=engine.graph_store,
        conversation_store=engine.conv_store,
        use_unified=True,
        use_adaptive_routing=use_adaptive,
        enable_temporal=True
    )
    print(f"\nQuestion: {args.query}")
    print("=" * 60)
    print(f"Status:   {res.get('status')}")
    print(f"Answer:   {res.get('answer')}")
    if res.get("citations"):
        cites = [c.get("memory_id") for c in res["citations"]]
        print(f"Citations: {', '.join(cites)}")
    if res.get("limitations"):
        print(f"Notes:    {res.get('limitations')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Argument Parser Construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ivara",
        description="IVARA — Personal AI Memory Assistant & Long-Term Reasoning System"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask an evidence-grounded personal memory question")
    ask_parser.add_argument("query", type=str, help="Question to ask IVARA")
    ask_parser.add_argument("--adaptive", action="store_true", help="Enable adaptive multi-source routing")
    ask_parser.add_argument("-k", type=int, default=5, help="Number of evidence items to retrieve")
    ask_parser.add_argument("--min-score", type=float, default=0.30, help="Relevance score floor")
    ask_parser.add_argument("--model", type=str, default="gpt-oss", help="Model name")
    ask_parser.set_defaults(func=cmd_ask)

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start or resume interactive conversational assistant")
    chat_parser.add_argument("--id", dest="id", type=str, default=None, help="Resume an existing conversation ID")
    chat_parser.add_argument("--sandbox", "-s", action="store_true", help="Run with isolated temporary runtime storage (does not touch production data)")
    chat_parser.set_defaults(func=cmd_chat)

    # chats command group
    chats_parser = subparsers.add_parser("chats", help="Manage conversation sessions")
    chats_sub = chats_parser.add_subparsers(dest="chats_subcommand")

    # chats list
    c_list = chats_sub.add_parser("list", help="List stored conversation sessions")
    c_list.add_argument("--limit", type=int, default=20, help="Max conversations to display")
    c_list.add_argument("--offset", type=int, default=0, help="Pagination offset")
    c_list.set_defaults(func=cmd_chats_list)

    # chat open <id> (also accessible as chats open <id>)
    c_open = chats_sub.add_parser("open", help="Open and resume a conversation")
    c_open.add_argument("id", type=str, help="Conversation ID to open")
    c_open.set_defaults(func=cmd_chat_open)

    # chat delete <id> (also accessible as chats delete <id>)
    c_del = chats_sub.add_parser("delete", help="Delete a conversation")
    c_del.add_argument("id", type=str, help="Conversation ID to delete")
    c_del.add_argument("--purge-memories", action="store_true", help="Also purge long-term memories originated from this chat")
    c_del.set_defaults(func=cmd_chat_delete)

    # chats clear
    c_clr = chats_sub.add_parser("clear", help="Clear all conversations")
    c_clr.add_argument("--yes", "-y", action="store_true", help="Bypass interactive confirmation")
    c_clr.set_defaults(func=cmd_chats_clear)

    # memories command group
    mem_parser = subparsers.add_parser("memories", help="Inspect and manage long-term memories")
    mem_sub = mem_parser.add_subparsers(dest="memories_subcommand")

    # memories list
    m_list = mem_sub.add_parser("list", help="List stored memories")
    m_list.add_argument("--limit", type=int, default=20, help="Max memories to show")
    m_list.set_defaults(func=cmd_memories_list)

    # memories search <query>
    m_search = mem_sub.add_parser("search", help="Search memories by semantic similarity")
    m_search.add_argument("query", type=str, help="Search query text")
    m_search.add_argument("-k", type=int, default=5, help="Number of results")
    m_search.add_argument("--min-score", type=float, default=0.25, help="Minimum cosine score threshold")
    m_search.set_defaults(func=cmd_memories_search)

    # memories show <id>
    m_show = mem_sub.add_parser("show", help="Display details of a specific memory")
    m_show.add_argument("id", type=str, help="Memory ID (e.g. m001)")
    m_show.set_defaults(func=cmd_memories_show)

    # memories delete <id>
    m_del = mem_sub.add_parser("delete", help="Delete a memory and its graph representations")
    m_del.add_argument("id", type=str, help="Memory ID to delete")
    m_del.set_defaults(func=cmd_memories_delete)

    # memories clear
    m_clr = mem_sub.add_parser("clear", help="Clear all long-term memories")
    m_clr.add_argument("--yes", "-y", action="store_true", help="Bypass interactive confirmation")
    m_clr.set_defaults(func=cmd_memories_clear)

    # memories stats
    m_stats = mem_sub.add_parser("stats", help="Show system memory & graph statistics")
    m_stats.set_defaults(func=cmd_memories_stats)

    return parser


def main():
    # If invoked with no arguments, launch interactive chat directly
    if len(sys.argv) == 1:
        cmd_chat(argparse.Namespace(id=None))
        return

    # Handle alias `python main.py chat open <id>` or `python main.py chat delete <id>`
    # where user uses `chat` instead of `chats` for subcommands
    args_list = list(sys.argv[1:])
    if len(args_list) >= 2 and args_list[0] == "chat" and args_list[1] in ("open", "delete", "list", "clear"):
        args_list[0] = "chats"

    parser = build_parser()
    args = parser.parse_args(args_list)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
