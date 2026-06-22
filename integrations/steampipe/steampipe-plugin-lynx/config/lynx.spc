connection "lynx" {
  plugin = "local/lynx"

  # Base URL of the local Lynx API. Optional — defaults to the LYNX_API env var,
  # then http://127.0.0.1:8765. Start the API with:
  #   lynx manager ui --port 8765 --no-browser
  api_url = "http://127.0.0.1:8765"
}
