./build/bin/sd -v --mode img2img \
	--model models/sdxl_turbo_q8_0.gguf \
	--height 1024 --width 1024 \
	--init-img "images/4guys.png" \
	--prompt "4 guys" \
	--negative-prompt "black and white" \
	--strength 0.20 \
	--cfg-scale 1.8 \
	--steps 5 \
	-o /tmp/output.png



# ./build/bin/sd -v --mode img2img \
# 	--model models/sd21_turbo_q8_0.gguf \
# 	--height 1024 --width 1024 \
#  	--init-img "images/4guys.png" \
# 	--prompt "4 guys" \
# 	--negative-prompt "black and white" \
# 	--strength 0.20 \
# 	--cfg-scale 7.5 \
# 	--steps 25 \
# 	-o /tmp/output.png


# ./build/bin/sd --mode img2img \
# 	--model models/sd21_turbo_q8_0.gguf \
# 	--height 640 --width 512 \
# 	--init-img images/bag.png \
# 	--prompt 'a children bag' --negative-prompt 'black and white' \
# 	--strength 0.80 \
# 	--control-net models/sd21_canny_q8_0.gguf --control-image images/bag.png \
# 	--canny --control-strength 0.70 \
# 	--cfg-scale 7.5 --steps 25 \
# 	-o /tmp/output.png


# ./build/bin/sd --mode img2img \
# 	--model models/sd21_turbo_q8_0.gguf \
# 	--height 640 --width 512 \
# 	--init-img images/bag.png \
# 	--prompt 'a children bag' --negative-prompt 'black and white' \
# 	--strength 0.10 \
# 	--control-net models/sd21_canny_q8_0.gguf \
# 	--control-image images/bag_scribble.png \
# 	--control-strength 0.70 \
# 	--cfg-scale 7.5 --steps 25 \
# 	-o /tmp/output.png



display /tmp/output.png

