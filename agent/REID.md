# Parking Recognition Evaluation Context

This document outlines the structure and mechanics of the Parking evaluation project located at `/mnt/secondary/test/Parking_Test/testing`. It provides the foundational knowledge required to build an automated evaluation feature based on this folder structure.

## Folder Structure

The project directory consists of the following key components:

1.  **`test_data.json`**: This file acts as the base template for the request payload sent to the Parking Recognition API. It contains fields for metadata (`id`, `gate_id`, `timestamp`), request parameters (e.g., confidence thresholds in the `param` object), and placeholders for image data (`camera.license_plate.image`, `camera.vehicle.image`) and the resulting OCR/classification `data`.
2.  **`test_case/`**: A directory containing individual JSON files (e.g., `test_case1.json`, `test_case2.json`). Each file represents a single evaluation scenario. A standard test case file defines:
    *   `name`: A descriptive name for the test (e.g., "1 Normal Mobil").
    *   `topic`: The event context (e.g., "IN").
    *   `lp_image`: The filename of the license plate image to use for this test.
    *   `vh_image`: The filename of the corresponding vehicle image.
    *   `expected`: A dictionary specifying the expected output from the system, including `license_plate` (value, confidence, status) and `vehicle` (type, subtype). It may also include assertion logic (e.g., checking if the vehicle crop image path matches a specific pattern).
3.  **`images/`**: A directory containing all the actual `.jpg` image assets referenced by the `lp_image` and `vh_image` fields in the test case files. It contains varied scenarios like normal cars, motorcycles, ambulances, vehicles with missing plates, and overlapping vehicles to robustly test the recognition system.

## How it Works

The intended workflow for the evaluation system is as follows:

1.  **Iterative Processing**: The evaluation runner iterates through every JSON file within the `test_case/` directory.
2.  **Payload Construction**: For a given test case, the runner loads the base `test_data.json` template. It reads the corresponding `lp_image` and `vh_image` from the `images/` directory and injects them (likely as base64 encoded strings or file uploads) into the payload. It also dynamically generates required metadata (like a unique `id` and current `timestamp`).
3.  **API Interaction**: The constructed payload is sent to the target Parking Recognition API endpoint.
4.  **Assertion and Validation**: The response from the API is parsed and compared against the `expected` block defined in the test case JSON. 
    *   It checks for exact matches on fields like `license_plate.value` and `vehicle.type`.
    *   It evaluates complex conditions, such as ensuring `confidence` is `>0.9` or validating dynamic string formats for image storage paths.

## How to Make It Work (Implementing the Evaluation Feature)

To create an automated evaluation script or feature based on this folder, you will need to implement the following steps:

1.  **Test Case Loader**: Write a function to scan the `test_case/` directory and load all `.json` files into memory as a list of dictionaries.
2.  **Image Handling**: Implement a helper function to read image files from the `images/` directory. If the API expects base64 encoding, convert the images accordingly.
3.  **Payload Builder**: Create a builder function that takes a test case dictionary, the base `test_data.json` template, and the loaded images to assemble a complete, valid request payload. Ensure you generate dynamic variables (timestamps, IDs) as needed.
4.  **API Client**: Implement the logic to send HTTP requests to the model/API endpoint being evaluated.
5.  **Evaluation Engine**: 
    *   Compare the actual API response against the `expected` dictionary.
    *   Implement logic to handle operators like `>0.9` (parse the operator and value and perform the mathematical comparison).
    *   Safely execute or evaluate dynamic assertions (like the f-string paths seen in the vehicle value expectation).
6.  **Reporting**: Aggregate the results (Pass/Fail) for all test cases and output a summary report, detailing any discrepancies between the expected and actual results.
