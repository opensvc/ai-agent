package config

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

// DocsConfig holds the documentation source shared by the backends.
type DocsConfig struct {
	// URL is the root of the book the backends search, or its llms.txt;
	// empty, they have no documentation search.
	URL string
}

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
	return cfg, nil
}
