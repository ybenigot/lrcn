for f in $CAFFE_HOME/examples/coco_caption/h5_data/*/*/image_list*.txt
do
	cp $f $f.old
	sed -e "1,\$s/\/media\/yves\/sandisk3\/caffe-master/\/Users\/yves\/Documents\/caffe2/" $f > $f.new
	cp $f.new $f
done

for f in $CAFFE_HOME/examples/coco_caption/h5_data/*/*/hdf5*.txt
do
	cp $f $f.old
	sed -e "1,\$s/\/media\/yves\/sandisk3\/caffe-master/\/Users\/yves\/Documents\/caffe2/" $f > $f.new
	cp $f.new $f	
done
