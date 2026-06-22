package lynx

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
)

const defaultAPIURL = "http://127.0.0.1:8765"

// apiBaseURL resolves the Lynx API base URL: connection config api_url, then the
// LYNX_API env var, then the local default. Trailing slash is trimmed.
func apiBaseURL(d *plugin.QueryData) string {
	cfg := GetConfig(d.Connection)
	if cfg.APIURL != nil && strings.TrimSpace(*cfg.APIURL) != "" {
		return strings.TrimRight(strings.TrimSpace(*cfg.APIURL), "/")
	}
	if env := strings.TrimSpace(os.Getenv("LYNX_API")); env != "" {
		return strings.TrimRight(env, "/")
	}
	return defaultAPIURL
}

// getNDJSON GETs <base>/api/v1/<path> with format=ndjson, decodes one JSON object
// per line, and passes each to emit. emit returns false to stop early (used to
// respect Steampipe's row limit). A 404 (e.g. /graph with no graph-enabled
// source) is treated as an empty result, not an error.
func getNDJSON(
	ctx context.Context,
	d *plugin.QueryData,
	path string,
	params url.Values,
	emit func(map[string]interface{}) bool,
) error {
	if params == nil {
		params = url.Values{}
	}
	params.Set("format", "ndjson")
	endpoint := fmt.Sprintf("%s/api/v1/%s?%s", apiBaseURL(d), path, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("lynx: request to %s failed (is `lynx manager ui` running?): %w", endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("lynx: %s returned %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(resp.Body)
	// Chunk content can be large (whole functions) — allow long lines.
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row map[string]interface{}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			// Skip a malformed line rather than abort the whole query.
			continue
		}
		if !emit(row) {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("lynx: error reading response from %s: %w", endpoint, err)
	}
	return nil
}
