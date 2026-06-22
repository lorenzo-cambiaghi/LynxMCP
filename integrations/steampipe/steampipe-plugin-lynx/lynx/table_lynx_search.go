package lynx

import (
	"context"
	"net/url"
	"strconv"

	"github.com/turbot/steampipe-plugin-sdk/v5/grpc/proto"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin/transform"
)

func tableLynxSearch(_ context.Context) *plugin.Table {
	return &plugin.Table{
		Name:        "lynx_search",
		Description: "Semantic + lexical code search over Lynx sources. The `query` qual is required; `source` and `top_k` are optional.",
		List: &plugin.ListConfig{
			KeyColumns: plugin.KeyColumnSlice{
				{Name: "query", Require: plugin.Required},
				{Name: "source", Require: plugin.Optional},
				{Name: "top_k", Require: plugin.Optional},
			},
			Hydrate: listLynxSearch,
		},
		Columns: []*plugin.Column{
			// Quals echoed back from the WHERE clause.
			{Name: "query", Type: proto.ColumnType_STRING, Transform: transform.FromQual("query"), Description: "Search query (required qual). Phrase it behaviourally — what the code does."},
			{Name: "top_k", Type: proto.ColumnType_INT, Transform: transform.FromQual("top_k"), Description: "Max hits to fetch (optional qual; server clamps to [1,50], default 8)."},
			// Result columns — one row per hit.
			{Name: "source", Type: proto.ColumnType_STRING, Transform: transform.FromField("source"), Description: "Source the hit came from (omit the qual to search every source, RRF-fused)."},
			{Name: "file", Type: proto.ColumnType_STRING, Transform: transform.FromField("file"), Description: "File name."},
			{Name: "file_path", Type: proto.ColumnType_STRING, Transform: transform.FromField("file_path"), Description: "Full file path."},
			{Name: "symbol", Type: proto.ColumnType_STRING, Transform: transform.FromField("symbol"), Description: "Qualified symbol name."},
			{Name: "kind", Type: proto.ColumnType_STRING, Transform: transform.FromField("kind"), Description: "Chunk kind (function, class, method, ...)."},
			{Name: "language", Type: proto.ColumnType_STRING, Transform: transform.FromField("language"), Description: "Programming language."},
			{Name: "start_line", Type: proto.ColumnType_INT, Transform: transform.FromField("start_line"), Description: "Start line of the chunk."},
			{Name: "end_line", Type: proto.ColumnType_INT, Transform: transform.FromField("end_line"), Description: "End line of the chunk."},
			{Name: "score", Type: proto.ColumnType_DOUBLE, Transform: transform.FromField("score"), Description: "Relevance score (RRF-fused dense + lexical)."},
			{Name: "content", Type: proto.ColumnType_STRING, Transform: transform.FromField("content"), Description: "The chunk content."},
		},
	}
}

func listLynxSearch(ctx context.Context, d *plugin.QueryData, _ *plugin.HydrateData) (interface{}, error) {
	params := url.Values{}
	params.Set("q", d.EqualsQualString("query"))
	if s := d.EqualsQualString("source"); s != "" {
		params.Set("source", s)
	}
	if q := d.EqualsQuals["top_k"]; q != nil {
		params.Set("top_k", strconv.FormatInt(q.GetInt64Value(), 10))
	}

	err := getNDJSON(ctx, d, "search", params, func(row map[string]interface{}) bool {
		d.StreamListItem(ctx, row)
		return d.RowsRemaining(ctx) > 0
	})
	return nil, err
}
