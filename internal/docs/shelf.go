package docs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// shelfRetry is how long a book that could not be read is not asked
	// again: a release line with no book published is asked once, not at
	// each search.
	shelfRetry = 10 * time.Minute
	// shelfLoadTimeout bounds the reading of the book of an earlier line,
	// the search waiting on it.
	shelfLoadTimeout = 30 * time.Second
	// shelfRefresh is how often the shelf reads again the versions of the
	// book and the index of the latest one: a new release line, or a new
	// build of the latest book, is seen within it.
	shelfRefresh = 15 * time.Minute
	// shelfUnreadRetry is how often the shelf tries to read a documentation
	// it could never read, such as one down when the agent started.
	shelfUnreadRetry = time.Minute
	// shelfRefreshTimeout bounds a refresh, the latest book read again
	// included.
	shelfRefreshTimeout = 2 * time.Minute
	// maxVersionsBytes bounds versions.json, and maxVersions its entries.
	maxVersionsBytes = 64 << 10
	maxVersions      = 64
)

// releaseLine is the major and minor numbers of a version, 3.0 of
// v3.0.0-rc56-41-g451250d44 or of "OpenSVC agent v3.0.0-rc56".
var releaseLine = regexp.MustCompile(`(?:^|[^0-9.])v?(\d{1,4})\.(\d{1,4})(?:[^0-9]|$)`)

// Line is the release line of a version, such as 3.0, empty when it names
// none.
func Line(version string) string {
	match := releaseLine.FindStringSubmatch(version)
	if match == nil {
		return ""
	}
	return match[1] + "." + match[2]
}

// fullVersion is a version as git describe gives it: v3.0.0, v3.0.0-rc56,
// or v3.0.0-rc56-41-g451250d44, 41 commits past the tag rc56.
var fullVersion = regexp.MustCompile(`^v?(\d{1,4})\.(\d{1,4})(?:\.(\d{1,6}))?(?:-([0-9A-Za-z.]+?))??(?:-(\d{1,9})-g[0-9a-f]{4,40})?$`)

// prerelease is a pre-release, such as rc56: its name, then its number.
var prerelease = regexp.MustCompile(`^([A-Za-z.]*)(\d*)$`)

// OlderVersion tells whether version a is older than b: by release line,
// then patch, then pre-release, a release coming after its pre-releases,
// then the commits git describe counts past the tag. A version of an
// unknown form is compared by its line only.
func OlderVersion(a, b string) bool {
	ma, mb := fullVersion.FindStringSubmatch(a), fullVersion.FindStringSubmatch(b)
	if ma == nil || mb == nil {
		return OlderLine(Line(a), Line(b))
	}
	number := func(s string) int {
		n, _ := strconv.Atoi(s)
		return n
	}
	for _, i := range []int{1, 2, 3} {
		if x, y := number(ma[i]), number(mb[i]); x != y {
			return x < y
		}
	}
	if pa, pb := ma[4], mb[4]; pa != pb {
		switch {
		case pa == "":
			return false
		case pb == "":
			return true
		}
		qa, qb := prerelease.FindStringSubmatch(pa), prerelease.FindStringSubmatch(pb)
		if qa == nil || qb == nil || qa[1] != qb[1] {
			return pa < pb
		}
		return number(qa[2]) < number(qb[2])
	}
	return number(ma[5]) < number(mb[5])
}

// OlderLine tells whether release line a is older than b.
func OlderLine(a, b string) bool {
	am, an := lineNumbers(a)
	bm, bn := lineNumbers(b)
	return am < bm || am == bm && an < bn
}

func lineNumbers(line string) (int, int) {
	match := releaseLine.FindStringSubmatch(line)
	if match == nil {
		return 0, 0
	}
	major, _ := strconv.Atoi(match[1])
	minor, _ := strconv.Atoi(match[2])
	return major, minor
}

// Shelf holds the books of a documentation published by release line, as
// its versions.json lists them for the release picker of the book:
//
//	[{"name": "v3.1", "path": "/"}, {"name": "v3.0", "path": "/v3.0/"}]
//
// The book whose path is / is the latest; the book of an earlier line is
// read the first time a search asks for it. A path is under the root of the
// book, which may be a directory of its site. An entry whose name carries no
// release line, or whose path is not a directory under the root, is left
// out, said in the log, as is an entry at / after the first. With no
// versions.json, or none usable, or no entry at /, the root holds one book,
// of the line of its keyword reference. The shelf reads versions.json and
// the index of the latest book again every shelfRefresh, the latest book
// when its build changed; until the latest book is first read, every
// shelfUnreadRetry.
type Shelf struct {
	root   *url.URL
	client *http.Client
	load   Loader
	logf   func(format string, args ...any)
	now    func() time.Time

	mu sync.Mutex
	// latest is the book at the root, nil until it is first read, of the
	// release its keyword reference documents, built build; lines are the
	// release lines with a book, latest first, latestLine the first, and
	// paths the path of each.
	latest     *Library
	release    string
	build      string
	latestLine string
	lines      []string
	paths      map[string]string
	// versions is the versions.json last read, checked when the shelf last
	// read the documentation, or failed to.
	versions   string
	checked    time.Time
	refreshing bool
	books      map[string]*shelved
}

// shelved is the book of one earlier line, read once: done closes when the
// read ends, and a read that failed is retried after shelfRetry.
type shelved struct {
	done    chan struct{}
	library *Library
	err     error
	at      time.Time
}

// version is an entry of versions.json.
type version struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// NewShelf makes the shelf of the documentation at root, such as
// https://book.opensvc.com/, read with client and load, which Refresh reads
// first; logf says what versions.json lists that the shelf leaves out. A
// root naming an llms.txt is its directory.
func NewShelf(root string, client *http.Client, load Loader, logf func(format string, args ...any)) (*Shelf, error) {
	link, err := url.Parse(strings.TrimSuffix(root, "llms.txt"))
	if err != nil || link.Scheme != "https" || link.Host == "" {
		return nil, fmt.Errorf("documentation root %q is not an https URL", root)
	}
	if !strings.HasSuffix(link.Path, "/") {
		link.Path += "/"
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Shelf{root: link, client: client, load: load, logf: logf, now: time.Now, paths: map[string]string{}, books: map[string]*shelved{}}, nil
}

// Root is the root of the documentation.
func (s *Shelf) Root() string { return s.root.String() }

// Latest is the book of the latest release line, nil until it is read.
func (s *Shelf) Latest() *Library {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.latest
}

// Lines are the release lines with a book, the latest first.
func (s *Shelf) Lines() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.lines)
}

// Refresh reads versions.json and the index of the latest book, and the
// latest book when its build changed since the last read, or it was never
// read. A failure leaves the shelf as it was, to be read again later.
func (s *Shelf) Refresh(ctx context.Context) error {
	err := s.refresh(ctx)
	s.mu.Lock()
	s.checked = s.now()
	s.mu.Unlock()
	return err
}

func (s *Shelf) refresh(ctx context.Context) error {
	versions, err := s.readVersions(ctx)
	if err != nil {
		return err
	}
	index := s.root.ResolveReference(&url.URL{Path: "llms.txt"}).String()
	text, err := get(ctx, s.client, index, maxIndexBytes)
	if err != nil {
		return err
	}
	build := ""
	for _, line := range strings.Split(text, "\n") {
		if match := indexBuild.FindStringSubmatch(line); match != nil {
			build = match[1]
			break
		}
	}
	s.mu.Lock()
	latest, release, read := s.latest, s.release, s.latest != nil && build == s.build
	changed := versions != s.versions
	s.mu.Unlock()
	if read && !changed {
		return nil
	}
	if !read {
		if latest, release, err = s.read(ctx, index); err != nil {
			return err
		}
	}
	entries := s.entries(versions, changed)
	latestLine, lines, paths := "", []string(nil), map[string]string{}
	for _, v := range entries {
		line := Line(v.Name)
		if v.Path == "/" {
			latestLine = line
		}
		if _, taken := paths[line]; !taken {
			lines, paths[line] = append(lines, line), v.Path
		}
	}
	// With no book at /, the root holds the book of its keyword reference.
	if latestLine == "" {
		if latestLine = Line(release); latestLine != "" {
			if _, taken := paths[latestLine]; !taken {
				lines = append(lines, latestLine)
			}
			paths[latestLine] = "/"
		}
	}
	slices.SortFunc(lines, func(a, b string) int {
		switch {
		case OlderLine(b, a):
			return -1
		case OlderLine(a, b):
			return 1
		}
		return 0
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latest, s.release, s.build, s.versions = latest, release, build, versions
	s.latestLine, s.lines, s.paths = latestLine, lines, paths
	// The books of the earlier lines are read again from the paths now
	// listed, the first time a search asks for them.
	if changed {
		s.books = map[string]*shelved{}
	}
	return nil
}

// read reads and indexes the book of an index with the loader. A panic of
// the loader, such as on a page the indexing did not foresee, is an error:
// the refresh or the search that asked for the book goes on without it,
// rather than the process stopping.
func (s *Shelf) read(ctx context.Context, index string) (library *Library, release string, err error) {
	defer func() {
		if r := recover(); r != nil {
			library, release, err = nil, "", fmt.Errorf("index %s: panic: %v", index, r)
		}
	}()
	return s.load(ctx, index)
}

// readVersions reads versions.json: empty when the documentation publishes
// none.
func (s *Shelf) readVersions(ctx context.Context) (string, error) {
	link := s.root.ResolveReference(&url.URL{Path: "versions.json"}).String()
	raw, err := get(ctx, s.client, link, maxVersionsBytes)
	var status *statusError
	if errors.As(err, &status) && status.code == http.StatusNotFound {
		return "", nil
	}
	return raw, err
}

// entries are the versions of versions.json the shelf can serve, the others
// said in the log when the file is new: a name carrying a release line, a
// path that is a directory under the root, and one at / only.
func (s *Shelf) entries(raw string, changed bool) []version {
	if raw == "" {
		return nil
	}
	logf := s.logf
	if !changed {
		logf = func(string, ...any) {}
	}
	var listed []version
	if err := json.Unmarshal([]byte(raw), &listed); err != nil {
		logf("documentation: versions.json of %s is no list of versions, left out: %v", s.root, err)
		return nil
	}
	if len(listed) > maxVersions {
		logf("documentation: versions.json of %s lists %d versions, the first %d kept", s.root, len(listed), maxVersions)
		listed = listed[:maxVersions]
	}
	var kept []version
	root := false
	for _, v := range listed {
		switch {
		case Line(v.Name) == "" || !strings.HasPrefix(v.Path, "/") || !strings.HasSuffix(v.Path, "/") || strings.Contains(v.Path, ".."):
			logf("documentation: versions.json of %s: %q at %q is no release line under the root, left out", s.root, v.Name, v.Path)
		case v.Path == "/" && root:
			logf("documentation: versions.json of %s: %q is a second version at /, left out", s.root, v.Name)
		default:
			root = root || v.Path == "/"
			kept = append(kept, v)
		}
	}
	if !root {
		logf("documentation: versions.json of %s lists no version at /: the root is the book of its keyword reference", s.root)
	}
	return kept
}

// refreshDue starts a refresh in the background when the last one is older
// than shelfRefresh, or shelfUnreadRetry while the latest book was never
// read: a search never waits on it, and a failed one leaves the books as
// they are until the next.
func (s *Shelf) refreshDue() {
	s.mu.Lock()
	defer s.mu.Unlock()
	every := shelfRefresh
	if s.latest == nil {
		every = shelfUnreadRetry
	}
	if s.refreshing || s.now().Sub(s.checked) < every {
		return
	}
	s.refreshing = true
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), shelfRefreshTimeout)
		defer cancel()
		if err := s.Refresh(ctx); err != nil {
			s.logf("documentation: read %s: %v", s.root, err)
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.refreshing = false
	}()
}

// Book gives the book for version, such as 3.0 or v3.0.0-rc56, and what to
// tell of the choice: the book of its release line, the latest one at the
// root, an earlier one under its path, or the latest one when version names
// no line, its line has no book, or is newer than the latest one.
func (s *Shelf) Book(ctx context.Context, version string) (*Library, string) {
	s.refreshDue()
	s.mu.Lock()
	latest, latestLine, path := s.latest, s.latestLine, ""
	line := Line(version)
	if line != "" {
		path = s.paths[line]
	}
	s.mu.Unlock()
	switch {
	case latest == nil:
		return nil, "The documentation could not be read."
	case latestLine == "":
		return latest, ""
	case version == "":
		return latest, fmt.Sprintf("No release line was asked: this is the book of the latest one, %s.", latestLine)
	case line == "":
		return latest, fmt.Sprintf("%q names no release line: this is the book of the latest one, %s.", version, latestLine)
	case line == latestLine:
		return latest, fmt.Sprintf("This is the book of the %s release line, the one of %s.", line, version)
	case OlderLine(latestLine, line):
		return latest, fmt.Sprintf("%s is of a release line newer than the latest book, %s: it may have what this book does not describe.", version, latestLine)
	case path == "":
		return latest, fmt.Sprintf("No book of the %s release line, the one of %s, is published: this is the book of the latest line, %s, which may describe what %s does not have.", line, version, latestLine, version)
	}
	library, err := s.earlier(ctx, line, path)
	if err != nil {
		return latest, fmt.Sprintf("The book of the %s release line, the one of %s, could not be read: this is the book of the latest line, %s, which may describe what %s does not have.", line, version, latestLine, version)
	}
	return library, fmt.Sprintf("This is the book of the %s release line, the one of %s.", line, version)
}

// earlier reads the book of an earlier line once, the searches of that line
// waiting on the first read, and again after shelfRetry when it failed.
func (s *Shelf) earlier(ctx context.Context, line, path string) (*Library, error) {
	s.mu.Lock()
	entry, ok := s.books[line]
	if ok && entry.err != nil && s.now().Sub(entry.at) >= shelfRetry {
		ok = false
	}
	if !ok {
		entry = &shelved{done: make(chan struct{})}
		s.books[line] = entry
		index := s.root.ResolveReference(&url.URL{Path: strings.TrimPrefix(path, "/") + "llms.txt"}).String()
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), shelfLoadTimeout)
			defer cancel()
			library, _, err := s.read(ctx, index)
			s.mu.Lock()
			entry.library, entry.err, entry.at = library, err, s.now()
			s.mu.Unlock()
			close(entry.done)
		}()
	}
	s.mu.Unlock()
	select {
	case <-entry.done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return entry.library, entry.err
}
