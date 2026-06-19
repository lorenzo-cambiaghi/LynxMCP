package lynx

import (
	"context"
	"net/url"
	"strconv"

	"github.com/turbot/steampipe-plugin-sdk/v5/grpc/proto"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin/transform"
)

func tableLynxGraph(_ context.Context) *plugin.Table {
	return &plugin.Table{
		Name:        "lynx_graph",
		Description: "Code knowledge graph edges (callers, callees, inheritance, imports, neighbors). The `operation` and `symbol` quals are required.",
		List: &plugin.ListConfig{
			KeyColumns: plugin.KeyColumnSlice{
				{Name: "operation", Require: plugin.Required},
				{Name: "symbol", Require: plugin.Required},
				{Name: "source", Require: plugin.Optional},
				{Name: "relation_filter", Require: plugin.Optional},
				{Name: "depth", Require: plugin.Optional},
				{Name: "edge_limit", Require: plugin.Optional},
			},
			Hydrate: listLynxGraph,
		},
		Columns: []*plugin.Column{
			// Quals echoed back from the WHERE clause.
			{Name: "operation", Type: proto.ColumnType_STRING, Transform: transform.FromQual("operation"), Description: "Graph operation (required qual): callers|callees|subclasses|superclasses|imports|neighbors."},
			{Name: "symbol", Type: proto.ColumnType_STRING, Transform: transform.FromQual("symbol"), Description: "Symbol to pivot from (required qual; fuzzy, case-insensitive substring)."},
			{Name: "relation_filter", Type: proto.ColumnType_STRING, Transform: transform.FromQual("relation_filter"), Description: "For neighbors: restrict to one edge relation (optional qual)."},
			{Name: "depth", Type: proto.ColumnType_INT, Transform: transform.FromQual("depth"), Description: "For neighbors: traversal depth 1-6 (optional qual)."},
			{Name: "edge_limit", Type: proto.ColumnType_INT, Transform: transform.FromQual("edge_limit"), Description: "Max edges to fetch (optional qual; server clamps to 200)."},
			// Result columns — one flat edge per row.
			{Name: "relation", Type: proto.ColumnType_STRING, Transform: transform.FromField("relation"), Description: "Edge relation (calls, extends, implements, imports, ...)."},
			{Name: "base_kind", Type: proto.ColumnType_STRING, Transform: transform.FromField("base_kind"), Description: "Kind of the queried (base) symbol."},
			{Name: "confidence", Type: proto.ColumnType_STRING, Transform: transform.FromField("confidence"), Description: "Resolution confidence of the edge."},
			{Name: "module", Type: proto.ColumnType_STRING, Transform: transform.FromField("module"), Description: "Imported path (for imports operation)."},
			{Name: "from_symbol", Type: proto.ColumnType_STRING, Transform: transform.FromField("from_symbol"), Description: "Source endpoint symbol of the edge."},
			{Name: "from_kind", Type: proto.ColumnType_STRING, Transform: transform.FromField("from_kind"), Description: "Source endpoint kind."},
			{Name: "from_file", Type: proto.ColumnType_STRING, Transform: transform.FromField("from_file"), Description: "Source endpoint file."},
			{Name: "from_start_line", Type: proto.ColumnType_INT, Transform: transform.FromField("from_start_line"), Description: "Source endpoint start line."},
			{Name: "from_end_line", Type: proto.ColumnType_INT, Transform: transform.FromField("from_end_line"), Description: "Source endpoint end line."},
			{Name: "to_symbol", Type: proto.ColumnType_STRING, Transform: transform.FromField("to_symbol"), Description: "Target endpoint symbol of the edge."},
			{Name: "to_kind", Type: proto.ColumnType_STRING, Transform: transform.FromField("to_kind"), Description: "Target endpoint kind."},
			{Name: "to_file", Type: proto.ColumnType_STRING, Transform: transform.FromField("to_file"), Description: "Target endpoint file."},
			{Name: "to_start_line", Type: proto.ColumnType_INT, Transform: transform.FromField("to_start_line"), Description: "Target endpoint start line."},
			{Name: "to_end_line", Type: proto.ColumnType_INT, Transform: transform.FromField("to_end_line"), Description: "Target endpoint end line."},
			{Name: "call_site_file", Type: proto.ColumnType_STRING, Transform: transform.FromField("call_site_file"), Description: "File of the call site (for call edges)."},
			{Name: "call_site_line", Type: proto.ColumnType_INT, Transform: transform.FromField("call_site_line"), Description: "Line of the call site (for call edges)."},
		},
	}
}

func listLynxGraph(ctx context.Context, d *plugin.QueryData, _ *plugin.HydrateData) (interface{}, error) {
	params := url.Values{}
	params.Set("operation", d.EqualsQualString("operation"))
	params.Set("symbol", d.EqualsQualString("symbol"))
	if s := d.EqualsQualString("source"); s != "" {
		params.Set("source", s)
	}
	if r := d.EqualsQualString("relation_filter"); r != "" {
		params.Set("relation", r)
	}
	if q := d.EqualsQuals["depth"]; q != nil {
		params.Set("depth", strconv.FormatInt(q.GetInt64Value(), 10))
	}
	if q := d.EqualsQuals["edge_limit"]; q != nil {
		params.Set("limit", strconv.FormatInt(q.GetInt64Value(), 10))
	}

	err := getNDJSON(ctx, d, "graph", params, func(row map[string]interface{}) bool {
		d.StreamListItem(ctx, row)
		return d.RowsRemaining(ctx) > 0
	})
	return nil, err
}
