package docs

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const testIndex = `# OpenSVC

> Documents the purpose and usage of the OpenSVC resilient and scalable infrastructure tools.

Build: bfc92ae 2026-11-02.
Keyword reference: OpenSVC agent v3.0.0-rc56.

## Agent

- [Configure > Cluster Quorum](agent/configure.quorum.md): When a peer is flagged as stale by all heartbeats, the daemon assumes a split-brain.
- [Install](https://elsewhere.example/agent/install.md): another host, never read.

## HOWTOs

- [NFS](howtos/nfsha.md)

## Optional

- [svc > DEFAULT](agent.reference.keywords/svc/DEFAULT.md): The keywords of the DEFAULT sections of the svc configuration.
- [Keyword changes](agent.reference.keywords/changes.md): What each release of the agent added, removed and changed in the keywords.
`

// book serves an llms.txt index under /v3.0/ and the pages it names, the
// one of a missing page apart; reads counts the requests of each path.
func book(t *testing.T, index string, pages map[string]string) (*httptest.Server, map[string]int) {
	t.Helper()
	reads := map[string]int{}
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reads[r.URL.Path]++
		if r.URL.Path == "/v3.0/llms.txt" {
			_, _ = w.Write([]byte(index))
			return
		}
		page, ok := pages[strings.TrimPrefix(r.URL.Path, "/v3.0/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(page))
	}))
	t.Cleanup(server.Close)
	return server, reads
}

func TestFetchReadsThePagesTheIndexLists(t *testing.T) {
	server, reads := book(t, testIndex, map[string]string{
		"agent/configure.quorum.md":           "# Cluster Quorum\n\nFrozen nodes do not evaluate quorum.\n",
		"howtos/nfsha.md":                     "# NFS\n\nAn howto.\n",
		"agent.reference.keywords/changes.md": "# Keyword changes\n\n## v3.0.0-rc56\n\n### Added\n\n- `cluster/ai.agent_url`\n",
	})
	// An optional page is read when include picks it, as any other.
	got, err := Fetch(context.Background(), server.Client(), server.URL+"/v3.0/llms.txt", func(section, path string) bool {
		return section != "HOWTOs" && (section != "Optional" || path == "agent.reference.keywords/changes.md")
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Build != "bfc92ae 2026-11-02" || got.KeywordReference != "OpenSVC agent v3.0.0-rc56" {
		t.Errorf("build %q, keyword reference %q", got.Build, got.KeywordReference)
	}
	if len(got.Pages) != 2 || got.Pages[0].Path != "agent/configure.quorum.md" || !strings.Contains(got.Pages[0].Text, "Frozen nodes") ||
		got.Pages[1].Path != "agent.reference.keywords/changes.md" || !strings.Contains(got.Pages[1].Text, "ai.agent_url") {
		t.Fatalf("pages = %+v", got.Pages)
	}
	// The pages left out, optional or not, and those of another host are
	// never read.
	for path, n := range reads {
		if path != "/v3.0/llms.txt" && path != "/v3.0/agent/configure.quorum.md" && path != "/v3.0/agent.reference.keywords/changes.md" {
			t.Errorf("%s read %d times", path, n)
		}
	}
}

func TestFetchFailsWithAPageMissing(t *testing.T) {
	server, _ := book(t, testIndex, map[string]string{"agent/configure.quorum.md": "# Cluster Quorum\n"})
	_, err := Fetch(context.Background(), server.Client(), server.URL+"/v3.0/llms.txt", func(section, _ string) bool { return section != "Optional" })
	if err == nil || !strings.Contains(err.Error(), "howtos/nfsha.md: 404") {
		t.Fatalf("err = %v", err)
	}
}

func TestFetchRefusesAnIndexOfNothing(t *testing.T) {
	server, _ := book(t, testIndex, nil)
	for index, want := range map[string]string{
		"http://book.example/llms.txt": "is not an https URL",
		server.URL + "/v3.0/llms.txt":  "lists no page to read",
	} {
		_, err := Fetch(context.Background(), server.Client(), index, func(string, string) bool { return false })
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Fetch(%s) = %v, want %q", index, err, want)
		}
	}
}

func TestFetchRefusesARedirectToAnotherHost(t *testing.T) {
	elsewhere := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("# Elsewhere\n"))
	}))
	t.Cleanup(elsewhere.Close)
	redirecting := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v3.0/llms.txt" {
			_, _ = w.Write([]byte(testIndex))
			return
		}
		http.Redirect(w, r, elsewhere.URL+r.URL.Path, http.StatusFound)
	}))
	t.Cleanup(redirecting.Close)
	_, err := Fetch(context.Background(), redirecting.Client(), redirecting.URL+"/v3.0/llms.txt", func(_, path string) bool { return path == "agent/configure.quorum.md" })
	if err == nil || !strings.Contains(err.Error(), "redirect to") {
		t.Fatalf("err = %v, want a refused redirect", err)
	}
}

// TestFetchReadsThePagesAfterALongLine reads an index whose description is a
// line longer than the 64 KiB a bufio.Scanner holds: the pages listed after
// it are read, none dropped.
func TestFetchReadsThePagesAfterALongLine(t *testing.T) {
	index := "# OpenSVC\r\n\r\n> " + strings.Repeat("A long description. ", 4000) + "\r\n\r\n## Agent\r\n\r\n" +
		"- [Install](agent/install.md): How to install.\r\n- [Quorum](agent/configure.quorum.md): The quorum.\r\n"
	server, _ := book(t, index, map[string]string{
		"agent/install.md":          "# Install\n",
		"agent/configure.quorum.md": "# Cluster Quorum\n",
	})
	got, err := Fetch(context.Background(), server.Client(), server.URL+"/v3.0/llms.txt", func(section, _ string) bool { return section == "Agent" })
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Pages) != 2 || got.Pages[0].Path != "agent/install.md" || got.Pages[1].Path != "agent/configure.quorum.md" {
		t.Fatalf("pages = %+v", got.Pages)
	}
}
