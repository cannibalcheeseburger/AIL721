ZIP_NAME="Kashish_aib242289"

mkdir $ZIP_NAME
rsync -War --files-from='zipass_files.txt' ./ $ZIP_NAME
zip $ZIP_NAME.zip -r $ZIP_NAME
rm -rf $ZIP_NAME

