package config

import "testing"

func TestLoadDocsURL(t *testing.T) {
	for _, tc := range []struct {
		name, value, want string
		wantErr           bool
	}{
		{"unset", "", "", false},
		{"https", " https://book.opensvc.com/llms.txt ", "https://book.opensvc.com/llms.txt", false},
		{"http", "http://book.opensvc.com/llms.txt", "", true},
		{"no host", "https:///llms.txt", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := loadDocs(func(key string) string {
				if key == "OPENSVC_AI_DOCS_URL" {
					return tc.value
				}
				return ""
			})
			if tc.wantErr != (err != nil) || cfg.URL != tc.want {
				t.Fatalf("OPENSVC_AI_DOCS_URL=%q: %+v, %v", tc.value, cfg, err)
			}
		})
	}
}
