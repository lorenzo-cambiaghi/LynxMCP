package lynx

import (
	"github.com/turbot/steampipe-plugin-sdk/v5/plugin"
)

// lynxConfig is the per-connection config parsed from the .spc file.
//
//	connection "lynx" {
//	  plugin  = "local/lynx"
//	  api_url = "http://127.0.0.1:8765"
//	}
type lynxConfig struct {
	// Base URL of the local Lynx API. Optional; falls back to the LYNX_API env
	// var, then http://127.0.0.1:8765.
	APIURL *string `hcl:"api_url"`
}

// ConfigInstance returns a new, empty config struct for the SDK to populate.
func ConfigInstance() interface{} {
	return &lynxConfig{}
}

// GetConfig retrieves the parsed config for a connection (zero value if unset).
func GetConfig(connection *plugin.Connection) lynxConfig {
	if connection == nil || connection.Config == nil {
		return lynxConfig{}
	}
	config, _ := connection.Config.(lynxConfig)
	return config
}
