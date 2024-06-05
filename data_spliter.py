import pandas as pd
import os

def split_csv_by_class(input_file, output_folder, chunk_size=45, max_files_per_class=5):
    # Проверка существования входного файла
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не существует.")
        return

    # Проверка существования выходной папки
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    try:
        # Чтение исходного CSV файла
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # Получение уникальных классов из столбца 'label'
    unique_classes = df['label'].unique()
    
    for cls in unique_classes:
        class_df = df[df['label'] == cls]
        
        file_count = 0
        for i in range(0, len(class_df), chunk_size):
            chunk = class_df.iloc[i:i + chunk_size]
            
            # Используем 1 заголовок и 45 строк данных
            if len(chunk) == chunk_size and file_count < max_files_per_class:
                # Имя для каждого нового файла
                output_file = os.path.join(output_folder, f"class_{cls}_chunk_{file_count + 1}.csv")

                try:
                    # Запись части данных в новый CSV файл
                    chunk.to_csv(output_file, index=False, header=(True))
                    file_count += 1
                    print(f"Файл сохранен: {output_file}")
    
                except Exception as e:
                    print(f"Ошибка при сохранении файла: {output_file}, Ошибка: {e}")
            
            # Прекращаем, если достигнуто максимальное количество файлов на класс
            if file_count >= max_files_per_class:
                break

if __name__ == "__main__":
    input_file = 'C:\\Library\\Diplom\\ML\\Project\\harth\\S024.csv'
    output_folder = 'C:\\Library\\Diplom\\ML\\Project\\harth_splited'

    # Запуск функции разбивки
    split_csv_by_class(input_file, output_folder)
