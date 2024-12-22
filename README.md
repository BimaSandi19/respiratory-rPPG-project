# Respiratory and rPPG Signal Measurement Project

Proyek ini mengimplementasikan sistem pengukuran sinyal pernapasan secara real-time yang dikombinasikan dengan sistem pengukuran remote photoplethysmography (rPPG). Sistem ini memproses input video dari webcam untuk mengekstrak dan menampilkan sinyal pernapasan dan rPPG.

## Anggota Kelompok

| Nama Anggota         | NIM         | Username       |
|----------------------|-------------|----------------|
| Bima Setiawan Sandi  | 121140162   | BimaSandi19    |
| Bayu Agaluh Wijaya   | 121140097   | Bayuagw        |
| Fatkhan Azies Suffi  | 120140181   | 120140181      |

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

Proyek respiratory-rPPG memanfaatkan teknik penglihatan komputer untuk menganalisis input video dari webcam. Proyek ini mengekstrak sinyal pernapasan dan sinyal rPPG secara real-time, memberikan wawasan berharga tentang kondisi fisiologis seseorang. Proyek ini dirancang untuk tujuan pendidikan dan penelitian.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/BimaSandi19/respiratory-rPPG-project.git
   cd respiratory-rPPG-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

Ensure that your webcam is connected and accessible. The application will open a window displaying the video feed along with the extracted respiratory and rPPG signals.

## Dependencies

The project requires the following Python packages:

- OpenCV
- NumPy
- Matplotlib

These dependencies are listed in the `requirements.txt` file.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.