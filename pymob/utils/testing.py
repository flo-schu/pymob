import numpy as np

def assert_no_infs_in_dataset(dataset):
    infs = bool((dataset == np.inf).sum().to_array().any())
    if infs:
        raise ValueError("Dataset contained infinity values")

def assert_no_nans_in_dataset(dataset):
    nans = bool(dataset.isnull().sum().to_array().any())
    if nans:
        raise ValueError("Dataset contained NaN values")

def _compare_config(config_a, config_b, ignore_fields={}):
    sections_mismatch = {}
    for section in config_a.model_fields.keys():
        a_section = getattr(config_a, section)
        b_section = getattr(config_b, section)


        if a_section != b_section:
            fields_mismatch = {}

            for field in a_section.model_fields_set:
                if field in ignore_fields.get(section, []):
                    continue

                a_field = getattr(a_section, field)
                b_field = getattr(b_section, field, "NOT DEFINED")

                if a_field != b_field:
                    fields_mismatch.update(
                        {field: {"origin": a_field, "export": b_field}}
                    )

            if len(fields_mismatch) > 0:
                sections_mismatch.update({section:fields_mismatch})

        else:
            pass

    if len(sections_mismatch) > 0:
        raise AssertionError(
            f"Config files do not match: {sections_mismatch=}"
        )