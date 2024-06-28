# VIVE Ultimate Tracker Camera Device
First instal `Steam` and `SteamVR`.
Then modify `SteamVR`'s two configs, as `"enable": false` to `"enable": true` in `<Steam Directory>/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings`, and `"requireHmd": true` to `"requireHmd": false`, `"forcedDriver": ""` to `"forcedDriver": "null"` and `"activateMultipleDrivers": false` to `"activateMultipleDrivers": true` in `<Steam Directory>/steamapps/common/SteamVR/resources/settings/default.vrsettings`, to support for headless.
Then install `VIVE_Streaming_Hub` from <https://dl.vive.com/vshubpc>. Update it to preview version (refer to <https://discord.com/channels/946305475467706419/1178656939333402675>) with preview code `VIVEUTRCPreview`.
Connect the wireless dongle, open `VIVE_Streaming_Hub`, update the firmware. Pair the Ultimate Tracker with the `VIVE_Streaming_Hub`, update the tracker's firmware.
Set up the tracker by building the map, which will automatically open the `SteamVR`.
