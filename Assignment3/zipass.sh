ZIP_NAME="2024AIB2289-Kashish-Srivastava.zip"

mkdir -p "$ZIP_NAME"

rsync -War --delete --files-from='./zipass_files.txt' ./ "$ZIP_NAME/"
zip -r "$ZIP_NAME.zip" "$ZIP_NAME"

rm -rf "$ZIP_NAME"