import os
import struct
from pathlib import Path
import numpy as np
import pandas as pd

def unpack_bcm4366c0_word(val: int):
    """
    Decodes a 32-bit BCM4366c0 packed floating-point CSI representation:
      - Bits [31:30]: Unused / padding
      - Bit  [29]:    Real sign bit
      - Bits [28:18]: Real mantissa magnitude (11 bits)
      - Bit  [17]:    Imag sign bit
      - Bits [16:6]:  Imag mantissa magnitude (11 bits)
      - Bits [5:0]:   Signed 6-bit exponent
    """
    exp_bits = val & 0x3F
    exp = exp_bits - 64 if (exp_bits & 0x20) else exp_bits
    scale = 2.0 ** exp

    imag_mag = (val >> 6) & 0x7FF
    imag_sign = (val >> 17) & 0x01
    imag = (-float(imag_mag) if imag_sign else float(imag_mag)) * scale

    real_mag = (val >> 18) & 0x7FF
    real_sign = (val >> 29) & 0x01
    real = (-float(real_mag) if real_sign else float(real_mag)) * scale

    return np.sqrt(real * real + imag * imag)

def decode_bcm4366c0_payload(payload_bytes: bytes, num_subcarriers: int = 256):
    """Parses a Nexmon UDP payload and returns a 256-element amplitude vector."""
    if len(payload_bytes) >= 1042:
        raw_csi = payload_bytes[18 : 18 + num_subcarriers * 4]
    elif len(payload_bytes) >= num_subcarriers * 4:
        raw_csi = payload_bytes[: num_subcarriers * 4]
    else:
        return None

    words = struct.unpack(f"<{num_subcarriers}I", raw_csi)
    amplitudes = np.array([unpack_bcm4366c0_word(w) for w in words], dtype=np.float32)

    if np.all(amplitudes == 0) or np.isnan(amplitudes).any():
        return None
    return amplitudes

def parse_pcap(pcap_path: Path, num_subcarriers: int = 256):
    """Pure-Python binary PCAP parser; returns (N, 256) float32 matrix."""
    amplitudes = []
    with open(pcap_path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            return np.empty((0, num_subcarriers), dtype=np.float32)

        magic = global_header[:4]
        if magic in (b'\xa1\xb2\xc3\xd4', b'\xa1\xb2\x3c\x4d'):
            endian = '>'
        elif magic in (b'\xd4\xc3\xb2\xa1', b'\x4d\x3c\xb2\xa1'):
            endian = '<'
        else:
            return np.empty((0, num_subcarriers), dtype=np.float32)

        while True:
            pkt_hdr = f.read(16)
            if len(pkt_hdr) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", pkt_hdr)
            pkt_data = f.read(incl_len)
            if len(pkt_data) < incl_len:
                break

            # Ethernet (14) + IP (min 20) + UDP (8)
            if len(pkt_data) < 42:
                continue

            eth_type = struct.unpack("!H", pkt_data[12:14])[0]
            if eth_type != 0x0800:  # IPv4
                continue

            ip_header_len = (pkt_data[14] & 0x0F) * 4
            ip_proto = pkt_data[23]
            if ip_proto != 17:      # UDP
                continue

            udp_offset = 14 + ip_header_len
            udp_payload = pkt_data[udp_offset + 8:]
            amp = decode_bcm4366c0_payload(udp_payload, num_subcarriers)
            if amp is not None:
                amplitudes.append(amp)

    if not amplitudes:
        return np.empty((0, num_subcarriers), dtype=np.float32)
    return np.vstack(amplitudes)

def process_experiment2_folders(input_csidata_dir: str, output_base_dir: str):
    input_dir = Path(input_csidata_dir)
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .pcap files recursively across all subfolders
    pcap_files = list(input_dir.rglob("*.pcap"))
    print(f"Discovered {len(pcap_files)} total .pcap files under {input_csidata_dir}\n")

    column_names = [f"subcarrier_{i:03d}" for i in range(256)]
    manifest_records = []

    for idx, pcap_path in enumerate(pcap_files, 1):
        rel_parts = pcap_path.relative_to(input_dir).parts
        
        # Determine split (development vs validation) and activity label
        if "validation" in rel_parts:
            split = "validation"
            # Path layout: validation/<activity>/<file.pcap>
            val_idx = rel_parts.index("validation")
            activity = rel_parts[val_idx + 1] if len(rel_parts) > val_idx + 1 else "unknown"
        else:
            split = "development"
            # Path layout: <activity>/<file.pcap>
            activity = rel_parts[0]

        # Construct organized output folder: output_dir / split / activity
        target_dir = output_dir / split / activity
        target_dir.mkdir(parents=True, exist_ok=True)
        target_csv_path = target_dir / (pcap_path.stem + ".csv")

        # Parse capture to (N, 256) amplitude array
        amp_data = parse_pcap(pcap_path, num_subcarriers=256)
        n_rows = amp_data.shape[0]

        if n_rows > 0:
            df = pd.DataFrame(amp_data, columns=column_names)
            df.to_csv(target_csv_path, index=False)
            status = "OK"
        else:
            status = "EMPTY"

        # Record provenance entry
        manifest_records.append({
            "capture_id": pcap_path.stem,
            "split": split,
            "activity": activity,
            "rows": n_rows,
            "status": status,
            "original_pcap": str(pcap_path),
            "csv_path": str(target_csv_path) if n_rows > 0 else None
        })

        if idx % 100 == 0 or idx == len(pcap_files):
            print(f"[{idx:4d}/{len(pcap_files)}] Processed: {split:<11} | {activity:<10} | {pcap_path.name} ({n_rows} rows)")

    # Save manifest table
    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = output_dir / "room_a_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    print("\n" + "=" * 55)
    print("Room A Processing Summary:")
    print(manifest_df.groupby(["split", "activity"])["capture_id"].count())
    print("=" * 55)
    print(f"Manifest saved to: {manifest_csv.resolve()}")

if __name__ == "__main__":
    INPUT_PATH = r"D:/Ub_Hack/Experiment-2/csidata"
    OUTPUT_PATH = r"D:/Ub_Hack/hack_2/intermediate_csv/room_A"

    process_experiment2_folders(INPUT_PATH, OUTPUT_PATH)