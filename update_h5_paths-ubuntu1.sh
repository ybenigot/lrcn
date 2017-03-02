for f in /media/yves/sandisk3/caffe-master/examples/coco_caption/h5_data/*/*/image_list*.txt
do
	sed -i "s/^./\/media\/yves\/sandisk3\/caffe-master/" $f
done

for f in /media/yves/sandisk3/caffe-master/examples/coco_caption/h5_data/*/*/hdf5*.txt
do
	sed -i "s/^./\/media\/yves\/sandisk3\/caffe-master/" $f
done
