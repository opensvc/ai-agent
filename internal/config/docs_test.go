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

func TestLoadCodeURL(t *testing.T) {
	for _, tc := range []struct {
		name, value, want string
		wantErr           bool
	}{
		{"unset", "", "", false},
		{"codeload", " https://codeload.github.com/opensvc/om3/tar.gz/{ref} ", "https://codeload.github.com/opensvc/om3/tar.gz/{ref}", false},
		{"no ref", "https://codeload.github.com/opensvc/om3/tar.gz/main", "", true},
		{"http", "http://mirror.example/om3/{ref}.tar.gz", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := loadDocs(func(key string) string {
				if key == "OPENSVC_AI_CODE_URL" {
					return tc.value
				}
				return ""
			})
			if tc.wantErr != (err != nil) || cfg.CodeURL != tc.want {
				t.Fatalf("OPENSVC_AI_CODE_URL=%q: %+v, %v", tc.value, cfg, err)
			}
		})
	}
}

func TestLoadCodeRef(t *testing.T) {
	for value, want := range map[string]string{"": "main", " v3.0.0-rc57 ": "v3.0.0-rc57", "release/3.0": "release/3.0", "bad ref": "", "../main": ""} {
		cfg, err := loadDocs(func(key string) string {
			if key == "OPENSVC_AI_CODE_REF" {
				return value
			}
			return ""
		})
		if (want == "") != (err != nil) || cfg.CodeRef != want {
			t.Errorf("OPENSVC_AI_CODE_REF=%q: %+v, %v", value, cfg, err)
		}
	}
}
