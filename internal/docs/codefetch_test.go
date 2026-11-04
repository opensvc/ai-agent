package docs

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCodeRefOfAVersion(t *testing.T) {
	for version, want := range map[string][2]string{
		"v3.0.0-rc56-41-g451250d44": {"451250d44", "v3.0.0-rc56"},
		"v3.0.0-rc50-8-g89140":      {"89140", "v3.0.0-rc50"},
		"v3.0.4":                    {"v3.0.4", ""},
		"v3.0.0-rc56":               {"v3.0.0-rc56", ""},
		"451250d44f0a1b2c3d4e5f60718293a4b5c6d7e8": {"451250d44f0a1b2c3d4e5f60718293a4b5c6d7e8", ""},
		"":                 {"", ""},
		"latest":           {"", ""},
		"v3.0.0; rm -rf /": {"", ""},
	} {
		if ref, tag := CodeRef(version); ref != want[0] || tag != want[1] {
			t.Errorf("CodeRef(%q) = %q, %q; want %q, %q", version, ref, tag, want[0], want[1])
		}
	}
}

type entry struct {
	name, text string
	kind       byte
}

func archiveOf(t *testing.T, entries ...entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(zw)
	for _, e := range entries {
		header := &tar.Header{Name: e.name, Mode: 0o644, Size: int64(len(e.text)), Typeflag: e.kind}
		if e.kind == tar.TypeSymlink {
			header.Size, header.Linkname = 0, "/etc/passwd"
		}
		if e.kind == 0 {
			header.Typeflag = tar.TypeReg
		}
		if err := tw.WriteHeader(header); err != nil {
			t.Fatal(err)
		}
		if header.Size > 0 {
			tw.Write([]byte(e.text))
		}
	}
	tw.Close()
	zw.Close()
	return buf.Bytes()
}

// codeServer serves archives by path, a missing one 404, and /moved
// redirected to another host.
func codeServer(t *testing.T, archives map[string][]byte) *httptest.Server {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/om3/moved" {
			http.Redirect(w, r, "https://elsewhere.example/om3.tar.gz", http.StatusFound)
			return
		}
		data, ok := archives[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestFetchCodeReadsTheGoFilesOfTheArchive(t *testing.T) {
	big := "package big\n// " + strings.Repeat("x", maxSourceFileBytes) + "\n"
	server := codeServer(t, map[string][]byte{"/om3/451250d44": archiveOf(t,
		entry{name: "om3-451250d44/", kind: tar.TypeDir},
		entry{name: "om3-451250d44/daemon/nmon/speaker.go", text: speakerFile},
		entry{name: "om3-451250d44/daemon/nmon/speaker_test.go", text: "package nmon\n"},
		entry{name: "om3-451250d44/README.md", text: "# om3"},
		entry{name: "om3-451250d44/link.go", kind: tar.TypeSymlink},
		entry{name: "om3-451250d44/big.go", text: big},
		entry{name: "om3-451250d44/../escape.go", text: "package escape\n"},
	)})
	files, err := FetchCode(context.Background(), server.Client(), server.URL+"/om3/{ref}", "451250d44")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 || files[0].Path != "daemon/nmon/speaker.go" || files[0].Text != speakerFile {
		t.Errorf("files = %+v", files)
	}
	if _, err := FetchCode(context.Background(), server.Client(), server.URL+"/om3/{ref}", "0000000"); !errors.Is(err, errNotPublished) {
		t.Errorf("a ref not published: %v", err)
	}
	if _, err := FetchCode(context.Background(), server.Client(), server.URL+"/om3/moved", "x"); err == nil || !strings.Contains(err.Error(), "redirect to elsewhere.example refused") {
		t.Errorf("a redirect elsewhere: %v", err)
	}
	if _, err := FetchCode(context.Background(), http.DefaultClient, "http://mirror.example/{ref}", "x"); err == nil || !strings.Contains(err.Error(), "is not an https URL") {
		t.Errorf("an http mirror: %v", err)
	}
}

func TestFetchCodeRefusesAnArchiveOfTooManyEntries(t *testing.T) {
	var entries []entry
	for i := 0; i <= maxArchiveFiles; i++ {
		entries = append(entries, entry{name: "om3/d/", kind: tar.TypeDir})
	}
	server := codeServer(t, map[string][]byte{"/om3/v1.0": archiveOf(t, entries...)})
	if _, err := FetchCode(context.Background(), server.Client(), server.URL+"/om3/{ref}", "v1.0"); err == nil || !strings.Contains(err.Error(), "more than 8000 entries") {
		t.Errorf("err = %v", err)
	}
}

func TestBoundedReaderStopsPastItsLimit(t *testing.T) {
	read := func(text string) error {
		_, err := io.ReadAll(&boundedReader{r: strings.NewReader(text), left: 4, limit: 4})
		return err
	}
	if err := read("abcd"); err != nil {
		t.Errorf("4 bytes: %v", err)
	}
	if err := read("abcde"); err == nil || !strings.Contains(err.Error(), "more than 4 bytes") {
		t.Errorf("5 bytes: %v", err)
	}
}
