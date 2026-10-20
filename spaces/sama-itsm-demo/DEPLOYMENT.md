# ITSM demo CI/CD

GitHub Actions runs the repository tests and builds the application image for
`linux/arm64` on pull requests. A push to `main` builds and publishes the image
to GHCR with both a commit SHA tag and the `latest` tag. OCI checks for `latest`
with a systemd timer every five minutes and deploys it locally. No inbound SSH
connection from GitHub Actions is required.

The OCI update script waits for the container health check. If the new image
does not become healthy, it retags the previous local image as `latest` and
restarts the service with that image.

## One-time GitHub setup

No SSH secrets or GitHub deployment environment are needed. The workflow uses
the repository's `GITHUB_TOKEN` with `packages: write` to publish the image.

## One-time OCI setup

Create a GitHub personal access token (classic) with only `read:packages`
scope. Log in to GHCR as the account that runs Docker Compose. On this server,
that is currently `root`; run:

```bash
read -rsp "GHCR read token: " GHCR_READ_TOKEN
printf '\n'
printf '%s' "$GHCR_READ_TOKEN" | docker login ghcr.io -u vivek5200 --password-stdin
unset GHCR_READ_TOKEN
```

The private GHCR package can then be pulled by the server. GitHub documents
`read:packages` for pulling private images and `GITHUB_TOKEN` for publishing
from Actions. [GHCR authentication guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

After the first successful workflow run publishes the image, install the
server-side updater from the cloned repository. First update the checkout to
the commit that added these files:

```bash
cd /root/projects/sama
git pull --ff-only origin main
cd /root/projects/sama/spaces/sama-itsm-demo
install -m 0755 deploy/sama-itsm-update.sh /usr/local/sbin/sama-itsm-update
install -m 0644 deploy/sama-itsm-update.service /etc/systemd/system/sama-itsm-update.service
install -m 0644 deploy/sama-itsm-update.timer /etc/systemd/system/sama-itsm-update.timer
systemctl daemon-reload
systemctl enable --now sama-itsm-update.timer
systemctl start sama-itsm-update.service
```

If the checkout is in another directory, change `APP_DIR` in
`deploy/sama-itsm-update.service` before installing it. The systemd service runs
as root so it can access Docker and the root user's GHCR login. The app's
`hf-cache` Docker volume persists the Qwen model across image updates.

## Operations

```bash
systemctl status sama-itsm-update.timer
systemctl start sama-itsm-update.service
journalctl -u sama-itsm-update.service -n 100 --no-pager
docker compose -f /root/projects/sama/spaces/sama-itsm-demo/docker-compose.yml ps
```

The server needs outbound access to GHCR and Hugging Face. This deployment
method makes no change to inbound HTTP or SSH firewall rules.
