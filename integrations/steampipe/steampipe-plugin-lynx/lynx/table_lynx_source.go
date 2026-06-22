package lynx

import (
	"context"

	"github.com/turbot/steampipe-plugin-sdk/v5/grpc/proto"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin/transform"
)

func tableLynxSource(_ context.Context) *plugin.Table {
	return &plugin.Table{
		Name:        "lynx_source",
		Description: "Sources indexed by the local Lynx instance (codebase, webdoc, pdf).",
		List: &plugin.ListConfig{
			Hydrate: listLynxSource,
		},
		Columns: []*plugin.Column{
			{Name: "name", Type: proto.ColumnType_STRING, Transform: transform.FromField("name"), Description: "Source name."},
			{Name: "type", Type: proto.ColumnType_STRING, Transform: transform.FromField("type"), Description: "Source type: codebase | webdoc | pdf."},
			{Name: "location", Type: proto.ColumnType_STRING, Transform: transform.FromField("location"), Description: "Filesystem path or URL of the source."},
			{Name: "chunk_count", Type: proto.ColumnType_INT, Transform: transform.FromField("chunk_count"), Description: "Number of indexed chunks."},
			{Name: "last_update", Type: proto.ColumnType_TIMESTAMP, Transform: transform.FromField("last_update"), Description: "When the source index was last updated."},
		},
	}
}

func listLynxSource(ctx context.Context, d *plugin.QueryData, _ *plugin.HydrateData) (interface{}, error) {
	err := getNDJSON(ctx, d, "sources", nil, func(row map[string]interface{}) bool {
		d.StreamListItem(ctx, row)
		return d.RowsRemaining(ctx) > 0
	})
	return nil, err
}
