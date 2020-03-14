for k in 20 25 30 35 40 50 70 90 110 130 150 170 190 210; do
  spark-submit --conf spark.driver.memory=240g OnlyLinks_TrainLDA.py --k $k
done
