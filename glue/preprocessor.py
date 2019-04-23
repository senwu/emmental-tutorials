import os
import codecs
import logging

from task_config import SPLIT_MAPPING, INDEX_MAPPING, SKIPPING_HEADER_MAPPING, LABEL_MAPPING

logger = logging.getLogger(__name__)



DELIMITER = "\t"

def parse_tsv(data_dir, task_name, split, max_data_samples=None):
    sentences = []
    labels = []

    tsv_path = os.path.join(data_dir, task_name, SPLIT_MAPPING[task_name][split])
    with codecs.open(tsv_path, "r", "utf-8") as f:
        # Skip header if needed
        if SKIPPING_HEADER_MAPPING[task_name]:
            f.readline()

        rows = list(enumerate(f))

        # Truncate to max_data_samples        
        if max_data_samples:
            rows = rows[:max_data_samples]
        
        max_cloumns = len(rows[0][1].strip().split(DELIMITER))

        for idx, row in rows:
            row = row.strip().split(DELIMITER)
            
            if len(row) > max_cloumns:
                logger.warning("Row has more columns than expected, skip...")
                continue
            
            sent1_idx, sent2_idx, label_idx = INDEX_MAPPING[task_name][split]

            if  sent1_idx >= len(row) or sent2_idx >= len(row) or label_idx >= len(row):
                logger.warning("Data column doesn't match data columns, skip...")
                continue

            sent1 = row[sent1_idx]
            sent2 = row[sent2_idx] if sent2_idx >= 0 else None

            if label_idx >= 0:
                if LABEL_MAPPING[task_name] is not None:
                    label = LABEL_MAPPING[task_name][row[label_idx]] 
                else:
                    label = row[label_idx] 
            else:
                label = -1

            sentences.append([sent1] if sent2 is None else [sent1, sent2])
            labels.append(label)

    return sentences, labels

# Test purpose
if __name__ == "__main__":
    sentences, labels = parse_tsv("data", "RTE", "train", max_data_samples=5)
    print(sentences, labels)
