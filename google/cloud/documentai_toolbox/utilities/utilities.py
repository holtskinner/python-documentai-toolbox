# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Document AI utilities."""

from typing import List, Optional

from google.cloud import documentai

from google.cloud.documentai_toolbox import constants
from google.cloud.documentai_toolbox.wrappers.document import _get_storage_client


def create_batches(
    gcs_bucket_name: str,
    gcs_prefix: str,
    batch_size: Optional[int] = constants.BATCH_MAX_FILES,
) -> List[documentai.BatchDocumentsInputConfig]:
    """Create batches of documents in Cloud Storage to process with `batch_process_documents()`.

    Args:
        gcs_bucket_name (str):
            Required. The name of the gcs bucket.

            Format: `gs://bucket/optional_folder/target_folder/` where gcs_bucket_name=`bucket`.
        gcs_prefix (str):
            Required. The prefix of the json files in the `target_folder`

            Format: `gs://bucket/optional_folder/target_folder/` where gcs_prefix=`optional_folder/target_folder`.
        batch_size (Optional[int]):
            Optional. Size of each batch of documents. Default is `50`.

    Returns:
        List[documentai.BatchDocumentsInputConfig]:
            A list of `BatchDocumentsInputConfig`, each corresponding to one batch.
    """
    if batch_size > constants.BATCH_MAX_FILES:
        raise ValueError(
            f"Batch size must be less than {constants.BATCH_MAX_FILES}. You provided {batch_size}."
        )

    storage_client = _get_storage_client()
    blob_list = storage_client.list_blobs(gcs_bucket_name, prefix=gcs_prefix)
    batches: List[documentai.BatchDocumentsInputConfig] = []
    batch: List[documentai.GcsDocument] = []

    for blob in blob_list:
        # Skip Directories
        if blob.name.endswith("/"):
            continue

        if blob.content_type not in constants.VALID_MIME_TYPES:
            print(f"Skipping file {blob.name}. Invalid Mime Type {blob.content_type}.")
            continue

        if blob.size > constants.BATCH_MAX_FILE_SIZE:
            print(
                f"Skipping file {blob.name}. File size must be less than {constants.BATCH_MAX_FILE_SIZE} bytes. File size is {blob.size} bytes."
            )
            continue

        if len(batch) == batch_size:
            batches.append(
                documentai.BatchDocumentsInputConfig(
                    gcs_documents=documentai.GcsDocuments(documents=batch)
                )
            )
            batch = []

        batch.append(
            documentai.GcsDocument(
                gcs_uri=f"gs://{gcs_bucket_name}/{blob.name}",
                mime_type=blob.content_type,
            )
        )

    if batch != []:
        # Append the last batch, which could be less than `batch_size`
        batches.append(
            documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(documents=batch)
            )
        )

    return batches


def document_type_to_processor_type(document_type: str) -> str:
    """Map output from the Procurement & Lending Splitter/Classifier to a Specialized Processor type.

    Args:
        classifier_output (str):
            Required. The output of a splitter/classifier processor. `Entity.type_`.

    Returns:
        str:
            A Specialized Processor type that can process the given document type. Returns `None` if there is no matching processor.
    """
    document_map = {
        # Procurement Documents
        "air_travel_statement": "EXPENSE_PROCESSOR",
        "car_rental_statement": "EXPENSE_PROCESSOR",
        "credit_card_slip": "EXPENSE_PROCESSOR",
        "credit_note": "INVOICE_PROCESSOR",
        "debit_note": "INVOICE_PROCESSOR",
        "ground_transportation_statement": "EXPENSE_PROCESSOR",
        "hotel_statement": "EXPENSE_PROCESSOR",
        "invoice_statement": "INVOICE_PROCESSOR",
        "purchase_order": "PURCHASE_ORDER_PROCESSOR",
        "receipt_statement": "EXPENSE_PROCESSOR",
        "restaurant_statement": "EXPENSE_PROCESSOR",
        "utility_statement": "UTILITY_PROCESSOR",
        # Lending Documents
        "1003": "FORM_1003_PROCESSOR",
        "1003_2009": "FORM_1003_PROCESSOR",
        "1005_1996": "FORM_1005_PROCESSOR",
        "1040": "FORM_1040_PROCESSOR",
        "1040_2018": "FORM_1040_PROCESSOR",
        "1040_2019": "FORM_1040_PROCESSOR",
        "1040_2020": "FORM_1040_PROCESSOR",
        "1040_2021": "FORM_1040_PROCESSOR",
        "1040nr": "FORM_1040NR_PROCESSOR",
        "1040nr_2018": "FORM_1040NR_PROCESSOR",
        "1040nr_2019": "FORM_1040NR_PROCESSOR",
        "1040nr_2020": "FORM_1040NR_PROCESSOR",
        "1040nr_2021": "FORM_1040NR_PROCESSOR",
        "1040sc": "FORM_1040SCH_C_PROCESSOR",
        "1040sc_2018": "FORM_1040SCH_C_PROCESSOR",
        "1040sc_2019": "FORM_1040SCH_C_PROCESSOR",
        "1040sc_2020": "FORM_1040SCH_C_PROCESSOR",
        "1040sc_2021": "FORM_1040SCH_C_PROCESSOR",
        "1040sr": "FORM_1040SR_PROCESSOR",
        "1040sr_2018": "FORM_1040SR_PROCESSOR",
        "1040sr_2019": "FORM_1040SR_PROCESSOR",
        "1040sr_2020": "FORM_1040SR_PROCESSOR",
        "1040sr_2021": "FORM_1040SR_PROCESSOR",
        "1065": "FORM_1065_PROCESSOR",
        "1065_2018": "FORM_1065_PROCESSOR",
        "1065_2019": "FORM_1065_PROCESSOR",
        "1065_2020": "FORM_1065_PROCESSOR",
        "1065_2021": "FORM_1065_PROCESSOR",
        "1076_2016": "FORM_1076_PROCESSOR",
        "1099div": "FORM_1099DIV_PROCESSOR",
        "1099div_2018": "FORM_1099DIV_PROCESSOR",
        "1099div_2019": "FORM_1099DIV_PROCESSOR",
        "1099div_2020": "FORM_1099DIV_PROCESSOR",
        "1099div_2021": "FORM_1099DIV_PROCESSOR",
        "1099g": "FORM_1099G_PROCESSOR",
        "1099g_2018": "FORM_1099G_PROCESSOR",
        "1099g_2019": "FORM_1099G_PROCESSOR",
        "1099g_2020": "FORM_1099G_PROCESSOR",
        "1099g_2021": "FORM_1099G_PROCESSOR",
        "1099int": "FORM_1099INT_PROCESSOR",
        "1099int_2018": "FORM_1099INT_PROCESSOR",
        "1099int_2019": "FORM_1099INT_PROCESSOR",
        "1099int_2020": "FORM_1099INT_PROCESSOR",
        "1099int_2021": "FORM_1099INT_PROCESSOR",
        "1099misc": "FORM_1099MISC_PROCESSOR",
        "1099misc_2018": "FORM_1099MISC_PROCESSOR",
        "1099misc_2019": "FORM_1099MISC_PROCESSOR",
        "1099misc_2020": "FORM_1099MISC_PROCESSOR",
        "1099misc_2021": "FORM_1099MISC_PROCESSOR",
        "1099nec": "FORM_1099NEC_PROCESSOR",
        "1099nec_2018": "FORM_1099NEC_PROCESSOR",
        "1099nec_2019": "FORM_1099NEC_PROCESSOR",
        "1099nec_2020": "FORM_1099NEC_PROCESSOR",
        "1099nec_2021": "FORM_1099NEC_PROCESSOR",
        "1099r": "FORM_1099R_PROCESSOR",
        "1099r_2018": "FORM_1099R_PROCESSOR",
        "1099r_2019": "FORM_1099R_PROCESSOR",
        "1099r_2020": "FORM_1099R_PROCESSOR",
        "1099r_2021": "FORM_1099R_PROCESSOR",
        "1099sb": "FORM_1040SCH_B_PROCESSOR",
        "1099sb_2018": "FORM_1040SCH_B_PROCESSOR",
        "1099sb_2019": "FORM_1040SCH_B_PROCESSOR",
        "1099sb_2020": "FORM_1040SCH_B_PROCESSOR",
        "1099sb_2021": "FORM_1040SCH_B_PROCESSOR",
        "1099sd": "FORM_1040SCH_D_PROCESSOR",
        "1099sd_2018": "FORM_1040SCH_D_PROCESSOR",
        "1099sd_2019": "FORM_1040SCH_D_PROCESSOR",
        "1099sd_2020": "FORM_1040SCH_D_PROCESSOR",
        "1099sd_2021": "FORM_1040SCH_D_PROCESSOR",
        "1099se": "FORM_1040SCH_E_PROCESSOR",
        "1099se_2018": "FORM_1040SCH_E_PROCESSOR",
        "1099se_2019": "FORM_1040SCH_E_PROCESSOR",
        "1099se_2020": "FORM_1040SCH_E_PROCESSOR",
        "1099se_2021": "FORM_1040SCH_E_PROCESSOR",
        "1099ssa": "FORM_SSA1099_PROCESSOR",
        "1099ssa_2018": "FORM_SSA1099_PROCESSOR",
        "1099ssa_2019": "FORM_SSA1099_PROCESSOR",
        "1099ssa_2020": "FORM_SSA1099_PROCESSOR",
        "1099ssa_2021": "FORM_SSA1099_PROCESSOR",
        "1120": "FORM_1120_PROCESSOR",
        "1120_2018": "FORM_1120_PROCESSOR",
        "1120_2019": "FORM_1120_PROCESSOR",
        "1120_2020": "FORM_1120_PROCESSOR",
        "1120_2021": "FORM_1120_PROCESSOR",
        "1120s": "FORM_1120S_PROCESSOR",
        "1120s_2018": "FORM_1120S_PROCESSOR",
        "1120s_2019": "FORM_1120S_PROCESSOR",
        "1120s_2020": "FORM_1120S_PROCESSOR",
        "1120s_2021": "FORM_1120S_PROCESSOR",
        "1_4_Family_Rider_3170": "FORM_FAMILY_RIDER_PROCESSOR",
        "3108_Adjustable_Rate_Rider": "FORM_ADJUSTABLE_RIDER_PROCESSOR",
        "3140_Condominium_Rider": "FORM_CONDOMINIUM_RIDER_PROCESSOR",
        "3190_Balloon_Rider": "FORM_BALLOON_RIDER_PROCESSOR",
        "3890_Second_Home_Rider": "FORM_SECOND_HOME_RIDER_PROCESSOR",
        "4506_T": "FORM_4506T_PROCESSOR",
        "4506_T_2018": "FORM_4506T_PROCESSOR",
        "4506_T_2019": "FORM_4506T_PROCESSOR",
        "4506_T_2020": "FORM_4506T_PROCESSOR",
        "4506_T_2021": "FORM_4506T_PROCESSOR",
        "4506_T_EZ": "FORM_4506T_EZ_PROCESSOR",
        "4506_T_EZ_2018": "FORM_4506T_EZ_PROCESSOR",
        "4506_T_EZ_2019": "FORM_4506T_EZ_PROCESSOR",
        "4506_T_EZ_2020": "FORM_4506T_EZ_PROCESSOR",
        "4506_T_EZ_2021": "FORM_4506T_EZ_PROCESSOR",
        "account_statement_bank": "BANK_STATEMENT_PROCESSOR",
        "account_statement_investment_and_retirement": "RETIREMENT_INVESTMENT_STATEMENT_PROCESSOR",
        "dhs_flood_certification": "FORM_FLOOD_CERTIFICATE_PROCESSOR",
        "f11_12956_2017": "FORM_F11_12956_PROCESSOR",
        "hud_54114": "FORM_HUD54114_PROCESSOR",
        "hud_92051": "FORM_HUD92051_PROCESSOR",
        "hud_92541": "FORM_HUD92541_PROCESSOR",
        "hud_92544": "FORM_HUD92544_PROCESSOR",
        "hud_92800": "FORM_HUD92800_PROCESSOR",
        "hud_92900a": "FORM_HUD92900A_PROCESSOR",
        "hud_92900b": "FORM_HUD92900B_PROCESSOR",
        "hud_92900lt": "FORM_HUD92900LT_PROCESSOR",
        "hud_92900ws": "FORM_HUD92900WS_PROCESSOR",
        "mortgage_statements": "MORTGAGE_STATEMENT_PROCESSOR",
        "payslip": "PAYSTUB_PROCESSOR",
        "property_insurance": "PROPERTY_INSURANCE_PROCESSOR",
        "pud_rider": "FORM_PUD_RIDER_PROCESSOR",
        "revocable_trust_rider": "FORM_REVOCABLE_TRUST_RIDER_PROCESSOR",
        "ssa_89": "FORM_SSA89_PROCESSOR",
        "ssa_89_2018": "FORM_SSA89_PROCESSOR",
        "ssa_89_2019": "FORM_SSA89_PROCESSOR",
        "ssa_89_2020": "FORM_SSA89_PROCESSOR",
        "ssa_89_2021": "FORM_SSA89_PROCESSOR",
        "ucc_financing_statement": "FORM_UCC1_PROCESSOR",
        "usda_ad_3030": "FORM_USDA_CONDITIONAL_COMMITMENT_PROCESSOR",
        "vba_26_0551_2004": "FORM_VBA26_0551_PROCESSOR",
        "vba_26_8923_2021": "FORM_VBA26_8923_PROCESSOR",
        "w2": "FORM_W2_PROCESSOR",
        "w2_2018": "FORM_W2_PROCESSOR",
        "w2_2019": "FORM_W2_PROCESSOR",
        "w2_2020": "FORM_W2_PROCESSOR",
        "w2_2021": "FORM_W2_PROCESSOR",
        "w9": "FORM_W9_PROCESSOR",
        "w9_2017": "FORM_W9_PROCESSOR",
        "w9_2018": "FORM_W9_PROCESSOR",
        "w9_2019": "FORM_W9_PROCESSOR",
        "w9_2020": "FORM_W9_PROCESSOR",
        "w9_2021": "FORM_W9_PROCESSOR",
        # Identity Documents
        "us_driver_license": "US_DRIVER_LICENSE_PROCESSOR",
        "us_passport": "US_PASSPORT_PROCESSOR",
    }

    return document_map.get(document_type, None)
