STEAMPIPE_INSTALL_DIR ?= ~/.steampipe
BUILD_TAGS = netgo

# Build the plugin and install it as a local plugin, so a connection with
#   plugin = "local/lynx"
# resolves to it. Run on macOS/Linux (where the Steampipe engine runs).
install:
	go build -o $(STEAMPIPE_INSTALL_DIR)/plugins/local/lynx/lynx.plugin -tags "$(BUILD_TAGS)" .

# Compile-check only (works on Windows too — no engine needed).
build:
	go build ./...

vet:
	go vet ./...

.PHONY: install build vet
