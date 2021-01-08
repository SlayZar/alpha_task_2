# Решение второй задачи соревнования по анализу данных Alfa Battle 2.0

Решение основано на агрегации скоров моделей бустинга и нейросети.

boost_i - модель бустинга с номером i

NN_i    - модель нейросети с номером i

1 - Усредняем скоры бустингов - MEAN[boost_1; boost_2; boost_3]

2 - Переводим усреднённые скоры бустингов из п.1 в ранги

3 - Получаем скоры двух моделей нейросети и переводим их в ранги - RANK{NN_1} ; RANK{NN_2}  

4 - Усредняем ранговые скоры из пункта 2 и пункта 3 - MEAN[RANK{MEAN[boost1; boost2; boost3]}; RANK{NN_1}; RANK{NN_2}]
