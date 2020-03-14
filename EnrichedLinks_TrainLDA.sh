for k in 210 35 150 20 25 30 40 50 70 90 110 130 170 190; do
  spark-submit --conf spark.driver.memory=240g EnrichedLinks_TrainLDA.py --k $k
done