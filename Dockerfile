# The image of the OpenSVC AI agent: the binary alone on alpine, run as
# opensvc-ai, uid 995 and gid 983, the IDs the files and the sockets of the
# service are given, the same on every node.
# The base images are pinned by digest: a tag moved in the registry does not
# change a release. The Go of the build is the toolchain line of go.mod, which
# CI tests and checks for vulnerabilities with: change both together. It is
# newer than the go line, the language version, for the fixes of the standard
# library the binary carries, Go 1.25 getting no more of them.
FROM golang:1.26.11@sha256:957952ae74dc010608717144399a59a2c6e82ea03d63be1c7ff6577c5a135f93 AS builder

# VERSION is the version of the binary, from git describe; the git directory
# is not in the build context.
ARG VERSION

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -buildvcs=false \
        -ldflags="-s -w -X github.com/opensvc/ai-agent/internal/buildinfo.version=${VERSION}" \
        -o /out/opensvc-ai-agentd ./cmd/opensvc-ai-agentd

FROM alpine:3.22.4@sha256:310c62b5e7ca5b08167e4384c68db0fd2905dd9c7493756d356e893909057601
ARG BUILDTIME

RUN addgroup -S -g 983 opensvc-ai \
    && adduser -S -D -H -u 995 -G opensvc-ai -s /sbin/nologin opensvc-ai

COPY --from=builder /out/opensvc-ai-agentd /usr/bin/opensvc-ai-agentd

USER 995:983
ENTRYPOINT ["/usr/bin/opensvc-ai-agentd"]

LABEL \
    org.opencontainers.image.authors="OpenSVC SAS" \
    org.opencontainers.image.created="${BUILDTIME}" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.url="https://github.com/opensvc/ai-agent"
