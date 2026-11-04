package config

import (
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
)

// DocsConfig holds the documentation source shared by the backends.
type DocsConfig struct {
	// URL is the root of the book the backends search, or its llms.txt;
	// empty, they have no documentation search.
	URL string
	// CodeURL is the tar.gz archive of the source code of the daemon at a
	// git reference, {ref} in it replaced by the reference, such as
	// https://codeload.github.com/opensvc/om3/tar.gz/{ref}; empty, the om3
	// backend has no code search.
	CodeURL string
	// CodeRef is the git reference of the latest code, read when a search
	// names no version: main by default.
	CodeRef string
}

// codeRef is a branch, a tag or a commit.
var codeRef = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$`)

func LoadDocs() (DocsConfig, error) {
	return loadDocs(os.Getenv)
}

func loadDocs(getenv func(string) string) (DocsConfig, error) {
	cfg := DocsConfig{}
	if value := strings.TrimSpace(getenv("OPENSVC_AI_DOCS_URL")); value != "" {
		link, err := url.Parse(value)
		if err != nil || link.Scheme != "https" || link.Host == "" {
			return DocsConfig{}, fmt.Errorf("parse OPENSVC_AI_DOCS_URL: %q is not an https URL", value)
		}
		cfg.URL = link.String()
	}
	if value := strings.TrimSpace(getenv("OPENSVC_AI_CODE_URL")); value != "" {
		link, err := url.Parse(strings.ReplaceAll(value, "{ref}", "ref"))
		if err != nil || link.Scheme != "https" || link.Host == "" || !strings.Contains(value, "{ref}") {
			return DocsConfig{}, fmt.Errorf("parse OPENSVC_AI_CODE_URL: %q is not an https URL holding {ref}", value)
		}
		cfg.CodeURL = value
	}
	cfg.CodeRef = "main"
	if value := strings.TrimSpace(getenv("OPENSVC_AI_CODE_REF")); value != "" {
		if !codeRef.MatchString(value) {
			return DocsConfig{}, fmt.Errorf("parse OPENSVC_AI_CODE_REF: %q is no git reference", value)
		}
		cfg.CodeRef = value
	}
	return cfg, nil
}
