ZIP_NAME="Kashish_2024AIB2289_Srivastava"

mkdir -p "$ZIP_NAME"

rsync -War --delete --files-from='./zipass_files.txt' ./ "$ZIP_NAME/"
zip -r "$ZIP_NAME.zip" "$ZIP_NAME"

rm -rf "$ZIP_NAME"