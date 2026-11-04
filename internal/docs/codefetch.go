package docs

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
)

const (
	// maxArchiveBytes bounds the compressed archive of a source tree, om3
	// weighing some 3 MB; maxArchiveFiles its entries, some 3 500; and
	// maxSourceFileBytes a file read from it.
	maxArchiveBytes    = 32 << 20
	maxArchiveFiles    = 8000
	maxSourceFileBytes = 2 << 20
	// maxSourceBytes bounds the Go source read from an archive in all, om3
	// holding some 10 MB of it.
	maxSourceBytes = 64 << 20
)

var (
	// describeVersion is the version git describe gives a build past a
	// tag: v3.0.0-rc56-41-g451250d44, the tag, the commits since it and the
	// commit, which a collector may report cut short, such as g89140: the
	// commit not found, the tag is read.
	describeVersion = regexp.MustCompile(`^(v?\d+\.\d+[0-9A-Za-z.+~-]*?)-\d+-g([0-9a-f]{4,40})$`)
	// tagVersion is a tag, such as v3.0.0-rc56 or v3.0.4.
	tagVersion = regexp.MustCompile(`^v?\d+\.\d+[0-9A-Za-z.+~-]*$`)
	// commitVersion is the commit a build with no tag reports.
	commitVersion = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// CodeRef is the git reference of the code of version, such as the commit
// 451250d44 of v3.0.0-rc56-41-g451250d44, and the tag it is past, read when
// the commit is not published; empty when version names no code.
func CodeRef(version string) (ref, tag string) {
	switch {
	case describeVersion.MatchString(version):
		m := describeVersion.FindStringSubmatch(version)
		return m[2], m[1]
	case commitVersion.MatchString(version):
		return version, ""
	case tagVersion.MatchString(version):
		return version, ""
	}
	return "", ""
}

// errNotPublished is an archive the server does not have, such as the
// commit of a local build never pushed.
var errNotPublished = errors.New("not published")

// FetchCode reads the Go files of the archive of ref, a tar.gz served at
// template with {ref} replaced, such as
// https://codeload.github.com/opensvc/om3/tar.gz/{ref}: the files under the
// root directory of the archive, the Go files only, symbolic links and
// other entries left out, nothing written on disk.
func FetchCode(ctx context.Context, client *http.Client, template, ref string) ([]File, error) {
	link := strings.ReplaceAll(template, "{ref}", url.PathEscape(ref))
	base, err := url.Parse(link)
	if err != nil || base.Scheme != "https" || base.Host == "" {
		return nil, fmt.Errorf("code archive %q is not an https URL", link)
	}
	// The archive is read from the configured host only, as the book is.
	sameHost := *client
	sameHost.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if request.URL.Scheme != base.Scheme || request.URL.Host != base.Host {
			return fmt.Errorf("redirect to %s refused: the code is read from %s only", request.URL.Host, base.Host)
		}
		if len(via) >= 10 {
			return fmt.Errorf("stopped after 10 redirects")
		}
		return nil
	}
	client = &sameHost
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, link, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", link, err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("read %s: %w", link, errNotPublished)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("read %s: %s", link, response.Status)
	}
	body := &boundedReader{r: response.Body, left: maxArchiveBytes, limit: maxArchiveBytes}
	unzipped, err := gzip.NewReader(body)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", link, err)
	}
	archive := tar.NewReader(unzipped)
	var (
		files   []File
		entries int
		total   int
	)
	for {
		header, err := archive.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", link, err)
		}
		if entries++; entries > maxArchiveFiles {
			return nil, fmt.Errorf("read %s: more than %d entries", link, maxArchiveFiles)
		}
		if header.Typeflag != tar.TypeReg {
			continue
		}
		// The archive holds the tree under one root directory, such as
		// om3-451250d44.../: the paths of the files are under it.
		name := path.Clean(header.Name)
		_, rel, ok := strings.Cut(name, "/")
		if !ok || strings.HasPrefix(rel, "../") || path.IsAbs(rel) || !strings.HasSuffix(rel, ".go") || strings.HasSuffix(rel, "_test.go") {
			continue
		}
		if header.Size > maxSourceFileBytes {
			continue
		}
		data, err := io.ReadAll(io.LimitReader(archive, maxSourceFileBytes+1))
		if err != nil {
			return nil, fmt.Errorf("read %s: %s: %w", link, rel, err)
		}
		if len(data) > maxSourceFileBytes {
			continue
		}
		if total += len(data); total > maxSourceBytes {
			return nil, fmt.Errorf("read %s: more than %d bytes of Go source", link, maxSourceBytes)
		}
		files = append(files, File{Path: rel, Text: string(data)})
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("read %s: no Go source", link)
	}
	return files, nil
}

// boundedReader fails past limit bytes, rather than reading on.
type boundedReader struct {
	r     io.Reader
	left  int64
	limit int64
}

func (b *boundedReader) Read(p []byte) (int, error) {
	if b.left <= 0 {
		// One byte more tells an archive of exactly limit bytes, which is
		// read, from a longer one.
		var probe [1]byte
		if n, _ := b.r.Read(probe[:]); n == 0 {
			return 0, io.EOF
		}
		return 0, fmt.Errorf("archive of more than %d bytes", b.limit)
	}
	if int64(len(p)) > b.left {
		p = p[:b.left]
	}
	n, err := b.r.Read(p)
	b.left -= int64(n)
	return n, err
}
