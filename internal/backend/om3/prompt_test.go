package om3

import (
	"strings"
	"testing"
)

func TestSystemPromptGroundsToolArguments(t *testing.T) {
	for _, instruction := range []string{
		"Never guess object paths, node names, resource identifiers, or other tool arguments.",
		"Only use identifiers provided by the user or returned by successful tool results.",
		"Examples in tool descriptions or schemas are illustrative and are never discovered identifiers.",
		"Never infer an identifier from an object name, naming convention, example, or failed tool output.",
		"If a prerequisite discovery tool fails or does not return a required identifier, do not call any dependent tool, even when a likely value can be inferred; stop that diagnostic branch and report the uncertainty.",
		"When multiple independent read-only tool calls are needed and all their arguments are already known, request them in the same step. Wait for prerequisite results before requesting dependent calls.",
		"Tool results of earlier turns may be partial: a _history_compaction field lists what was omitted. Treat omitted data as unknown. When you need omitted details, call an appropriate read-only tool, never again a tool that changes state. A new read is a new observation that may differ from the earlier one: when the user asks for an earlier value that was omitted, say it is no longer available.",
		"In tool provenance, observed_at is when the MCP collected a result, not when OpenSVC updated the underlying status; use daemon-provided updated_at when available to assess status age.",
	} {
		if !strings.Contains(SystemPrompt, instruction) {
			t.Errorf("system prompt is missing instruction %q", instruction)
		}
	}
}

// An approval answers the question asked, on the target named: on
// 2026-10-23, a yes given after the work was done led a model to stop, then
// purge, the object whose storage of one node it had just freed.
func TestSystemPromptScopesApprovals(t *testing.T) {
	for _, instruction := range []string{
		"If your last message asked no question, a yes, go ahead, ok or any approval authorizes nothing: say the work is done and call no tool that changes state. An approval answers only the last question you asked.",
		"An action asked on one node acts on that node: it does not become an action on the object, nor on its other nodes. A stronger action than the one asked, a stop before an unprovision, a purge instead of an unprovision, is a new question to ask, never a step you add.",
		"Do not infer what a tool does from its name: read its description before deciding it needs a preparation step.",
	} {
		if !strings.Contains(SystemPrompt, instruction) {
			t.Errorf("system prompt is missing instruction %q", instruction)
		}
	}
}

// TestDocsPromptStandsApart keeps every word of search_docs out of the prompt
// of a backend with no book to search, where the tool is not offered.
func TestDocsPromptStandsApart(t *testing.T) {
	if strings.Contains(SystemPrompt, "search_docs") {
		t.Error("SystemPrompt names search_docs, offered only with a book")
	}
	for _, rule := range []string{"search_docs", "list_node_config_keywords", "list_object_config_keywords"} {
		if !strings.Contains(DocsPrompt, rule) {
			t.Errorf("DocsPrompt lacks %q", rule)
		}
	}
}
