package main

import (
	"github.com/lorenzo-cambiaghi/steampipe-plugin-lynx/lynx"
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
)

func main() {
	plugin.Serve(&plugin.ServeOpts{PluginFunc: lynx.Plugin})
}
