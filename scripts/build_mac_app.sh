#!/usr/bin/env zsh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="$HOME/Desktop/ARIES Public Preview Beta.app"
APP_RES="$APP_PATH/Contents/Resources"
BUNDLE_APP="$APP_RES/app"

rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS" "$BUNDLE_APP"

cat > "$APP_PATH/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>ARIES Public Preview Beta</string>
  <key>CFBundleExecutable</key><string>ARIES Public Preview Beta</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>org.aries.public-preview-beta</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>ARIES Public Preview Beta</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>ARIES uses your microphone for local speech recognition.</string>
</dict>
</plist>
PLIST

if [ -f "$REPO_DIR/static/AppIcon.icns" ]; then
  cp "$REPO_DIR/static/AppIcon.icns" "$APP_RES/AppIcon.icns"
fi

# Bundle app files inside Contents/Resources/app
cp "$REPO_DIR"/*.py "$BUNDLE_APP/"
cp "$REPO_DIR/LICENSES.md" "$REPO_DIR/requirements.txt" "$BUNDLE_APP/"
cp -R "$REPO_DIR/static" "$BUNDLE_APP/"
cp -a "$REPO_DIR/engine" "$BUNDLE_APP/"
if [ -e "$REPO_DIR/.venv" ]; then
  cp -a "$REPO_DIR/.venv" "$BUNDLE_APP/.venv"
fi

# Compile native arm64 Mach-O launcher binary (avoids macOS TCC platform-binary Desktop deny)
TMP_C="$(mktemp /tmp/aries_launcher_XXXXXX.c)"
cat > "$TMP_C" <<'CSRC'
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char exe[PATH_MAX];
    uint32_t sz = sizeof(exe);
    if (_NSGetExecutablePath(exe, &sz) != 0) return 1;

    char resolved[PATH_MAX];
    if (!realpath(exe, resolved)) return 1;

    char *slash = strrchr(resolved, '/');
    if (!slash) return 1;
    *slash = '\0'; /* .../Contents/MacOS */
    slash = strrchr(resolved, '/');
    if (!slash) return 1;
    *slash = '\0'; /* .../Contents */

    char app_dir[PATH_MAX];
    snprintf(app_dir, sizeof(app_dir), "%s/Resources/app", resolved);
    if (chdir(app_dir) != 0) return 2;

    system("lsof -ti :8776 -i :8777 -i :8778 | xargs kill -9 2>/dev/null || true");

    char py[PATH_MAX];
    snprintf(py, sizeof(py), "%s/.venv/bin/python", app_dir);
    char entry[PATH_MAX];
    snprintf(entry, sizeof(entry), "%s/web_app.py", app_dir);

    char *args[] = {py, entry, NULL};
    execv(py, args);
    return 3;
}
CSRC

clang -O2 -arch arm64 "$TMP_C" -o "$APP_PATH/Contents/MacOS/ARIES Public Preview Beta"
rm -f "$TMP_C"

xattr -cr "$APP_PATH"
codesign --force --deep --sign - "$APP_PATH"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_PATH"
echo "Built $APP_PATH"
