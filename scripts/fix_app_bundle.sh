#!/bin/bash
#
# Post-build fix for macOS app bundle
# Removes LSRequiresCarbon and adds PkgInfo
#

APP_BUNDLE="$1"

if [ ! -d "$APP_BUNDLE" ]; then
    echo "Error: App bundle not found: $APP_BUNDLE"
    exit 1
fi

echo "Fixing app bundle: $APP_BUNDLE"

# Remove LSRequiresCarbon from Info.plist
/usr/libexec/PlistBuddy -c "Delete LSRequiresCarbon" "$APP_BUNDLE/Contents/Info.plist" 2>&1

# Add PkgInfo file
printf "APPL????\n" > "$APP_BUNDLE/Contents/PkgInfo"
chmod 644 "$APP_BUNDLE/Contents/PkgInfo"

# Re-sign the app
codesign --remove-signature "$APP_BUNDLE" 2>/dev/null
codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null

echo "✓ App bundle fixed"
