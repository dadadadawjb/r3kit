# VIVE Tracker 3 Device
First instal `Steam` and `SteamVR`.
Then modify `SteamVR`'s two configs to support for headless:
* `<Steam Directory>/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings`: `"enable": false` to `"enable": true`
* `<Steam Directory>/steamapps/common/SteamVR/resources/settings/default.vrsettings`: `"requireHmd": true` to `"requireHmd": false`, `"forcedDriver": ""` to `"forcedDriver": "null"` and `"activateMultipleDrivers": false` to `"activateMultipleDrivers": true`

To change the action rule, `Manage trackers` in `SteamVR`.

## Coordinate
* Shared tracking map: unknown
* Tracker self: (x=left, y=up, z=in)
