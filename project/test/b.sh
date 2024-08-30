# ./build/bin/sd -v --mode img2img \
# 	--model models/sdxl_turbo_q8_0.gguf \
# 	--height 1024 --width 1024 \
# 	--init-img "images/4guys.png" \
# 	--prompt "4 guys standing in back of sea, sunset, four person face to camera !!!" \
# 	--negative-prompt "black and white" \
# 	--strength 0.15 \
# 	--cfg-scale 1.8 \
# 	--steps 5 --seed -1 \
# 	-o /tmp/output1.png

# display /tmp/output1.png

# PROMPT="interior design of a luxurious master bedroom, gold and marble furniture, luxury, intricate, breathtaking"
# NEGATIVE="black and white"
# ./build/bin/sd --mode txt2img --model models/sdxl_turbo_q8_0.gguf \
# 	--height 768 --width 512 \
# 	--prompt "'$PROMPT'" \
# 	--negative-prompt "'$NEGATIVE'" \
# 	--strength 0.90 \
# 	--cfg-scale 1.8 --steps 5 --seed -1 --threads 1 \
# 	-o /tmp/output2.png


# display /tmp/output2.png


PROMPT="high-resolution photograph of a translucent softly glowing mechanical human brain, visible internal organs, polycarbonate, and glass, very detailed, 8k, sharp, fantasy, movie poster, unreal engine 5"
NEGATIVE="blurry, blur, text, watermark, render, 3D, NSFW, nude, CGI, monochrome, B&W, cartoon, painting, smooth, plastic, blurry, low-resolution, deep-fried, oversaturated"
./build/bin/sd --mode txt2img --model models/sdxl_turbo_q8_0.gguf \
	--height 768 --width 768 \
	--prompt "'$PROMPT'" \
	--negative-prompt "'$NEGATIVE'" \
	--strength 0.90 \
	--cfg-scale 1.8 --steps 5 \
	-o /tmp/demo_ok.png