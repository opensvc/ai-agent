package llm

import (
	"encoding/json"
	"testing"
)

func TestPlainTextIsTheTextOfAResultOfTextAlone(t *testing.T) {
	for _, test := range []struct {
		content string
		isError bool
		want    string
		ok      bool
	}{
		{`{"content":[{"type":"text","text":"a\nb"}]}`, false, "a\nb", true},
		{`{"content":[{"type":"text","text":"a"}],"isError":false}`, false, "a", true},
		{`{"content":[{"type":"text","text":"{\"tools\":[]}"}]}`, false, `{"tools":[]}`, true},
		{`{"content":[{"type":"text","text":"a"}]}`, true, "", false},
		{`{"content":[{"type":"text","text":"a"}],"isError":true}`, false, "", false},
		{`{"content":[{"type":"text","text":"{}"}],"structuredContent":{}}`, false, "", false},
		{`{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`, false, "", false},
		{`{"content":[{"type":"text","text":"a","annotations":{}}]}`, false, "", false},
		{`{"content":[{"type":"text","text":"a","_meta":{}}]}`, false, "", false},
		{`{"content":[{"type":"text","text":""}]}`, false, "", false},
		{`{"content":[{"type":"text","text":null}]}`, false, "", false},
		{`{"content":[{"type":"image","data":"AA"}]}`, false, "", false},
		{`{"content":[],"_history_compaction":{}}`, false, "", false},
		{`not json`, false, "", false},
	} {
		got, ok := ToolResult{Content: json.RawMessage(test.content), IsError: test.isError}.PlainText()
		if got != test.want || ok != test.ok {
			t.Errorf("%s error=%v: %q %v, want %q %v", test.content, test.isError, got, ok, test.want, test.ok)
		}
	}
}
