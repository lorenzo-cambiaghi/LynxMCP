package lynx

import (
	"context"

	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin/transform"
)

// Plugin is the entry point Steampipe calls to build the plugin definition.
// Every table maps 1:1 onto a Lynx /api/v1 endpoint; the Go is a thin SQL skin
// over the local HTTP API (see integrations/steampipe/DESIGN.md).
func Plugin(ctx context.Context) *plugin.Plugin {
	return &plugin.Plugin{
		Name: "steampipe-plugin-lynx",
		ConnectionConfigSchema: &plugin.ConnectionConfigSchema{
			NewInstance: ConfigInstance,
		},
		// Columns extract from streamed map[string]interface{} rows via explicit
		// FromField/FromQual transforms, so the default is only a fallback.
		DefaultTransform: transform.FromGo().NullIfZero(),
		TableMap: map[string]*plugin.Table{
			"lynx_source": tableLynxSource(ctx),
			"lynx_search": tableLynxSearch(ctx),
			"lynx_graph":  tableLynxGraph(ctx),
		},
	}
}
