def filter_existing_fields(ds):
    def validate_row(row):
        if row["image"] is None:
            print("No image")
            return False
        try:
            with PILImage.open(row["image"]["path"]) as img:
                # print("No valid image")
                img.verify()
            return True
        except (UnidentifiedImageError, IOError):
            print("other issue")
            return False

    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(validate_row)
    return ds


def preprocess_dataset(ds):
    ds = ds.remove_columns(["file_name"])

    features = Features(
        {
            "image": Image(mode=None, decode=True, id=None),
            "id": Value(dtype="string", id=None),
            "kingdom_key": Value(dtype="uint32", id=None),
            "phylum_key": Value(dtype="uint32", id=None),
            "order_key": Value(dtype="uint32", id=None),
            "family_key": Value(dtype="uint32", id=None),
            "genus_key": Value(dtype="uint32", id=None),
            "scientific_name": Value(dtype="string", id=None),
            "species": Value(dtype="string", id=None),
            "sex": Value(dtype="string", id=None),
            "life_stage": Value(dtype="string", id=None),
            "continent": Value(dtype="string", id=None),
        }
    )

    ds = ds.cast(features)
    # ds = ds.map(preprocess_entry)

    def valid_image_shape(row):
        return row["image"].mode == "RGB"

    ds = ds.filter(valid_image_shape)
    ds_dict = datasets.DatasetDict({"data": ds})
    return ds_dict

