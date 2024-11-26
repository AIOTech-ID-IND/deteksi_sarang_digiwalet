async function detect_objects_on_image(buf, model, bulu_classes) {
    const buf_resized = await resizeImage(buf);

    const [squaredImageBuffer, ori_img_width, ori_img_height] = await addGreenPaddingToSquareImage(buf_resized);

    const [squaredInput, img_width, img_height] = await prepare_input(squaredImageBuffer);

    const output = await run_model(squaredInput, model);
 
    const result = process_output(output, img_width, img_height, bulu_classes);

    const outputImageBuffer_req = await draw_boxes_on_image(squaredImageBuffer, result);

    const outputImageBuffer = await removeGreenPadding(outputImageBuffer_req, ori_img_width, ori_img_height);
  
    return [outputImageBuffer, result];
  }
  
  async function resizeImage(imageBuffer) {
    const imgBitmap = await createImageBitmap(new Blob([imageBuffer]));
  
    const aspectRatio = imgBitmap.width / imgBitmap.height;
  
    let newWidth, newHeight;
    if (imgBitmap.width > imgBitmap.height) {
      newWidth = 640;
      newHeight = 640 / aspectRatio;
    } else {
      newHeight = 640;
      newWidth = 640 * aspectRatio;
    }
  
    const canvas = new OffscreenCanvas(newWidth, newHeight);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(imgBitmap, 0, 0, newWidth, newHeight);
  
    const blob = await canvas.convertToBlob({ type: 'image/jpeg' });
    return blob;
  }
  
  async function addGreenPaddingToSquareImage(imageBuffer) {
    const imgBitmap = await createImageBitmap(new Blob([imageBuffer]));
    
    const size = Math.max(imgBitmap.width, imgBitmap.height);
    
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'rgb(0, 255, 0)';
    ctx.fillRect(0, 0, size, size);

    const offsetX = (size - imgBitmap.width) / 2;
    const offsetY = (size - imgBitmap.height) / 2;
    
    ctx.drawImage(imgBitmap, offsetX, offsetY);
    
    const blob = await canvas.convertToBlob({ type: 'image/jpeg' });
    return [blob, imgBitmap.width, imgBitmap.height];
  }
  
  
  async function removeGreenPadding(squaredImageBuffer, originalWidth, originalHeight) {
    const imgBitmap = await createImageBitmap(new Blob([squaredImageBuffer]));
    const size = Math.max(imgBitmap.width, imgBitmap.height);
    const offsetX = (size - originalWidth) / 2;
    const offsetY = (size - originalHeight) / 2;
  
    const canvas = new OffscreenCanvas(originalWidth, originalHeight);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgBitmap, offsetX, offsetY, originalWidth, originalHeight, 0, 0, originalWidth, originalHeight);
  
    return await canvas.convertToBlob({ type: 'image/jpeg' });
  }
  
  async function prepare_input(buf) {
    const imgBitmap = await createImageBitmap(new Blob([buf]));
    const img_width = imgBitmap.width;
    const img_height = imgBitmap.height;
  
    const canvas = new OffscreenCanvas(640, 640);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgBitmap, 0, 0, img_width, img_height, 0, 0, 640, 640);
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const pixels = imageData.data;
  
    const input = new Float32Array(640 * 640 * 3);
    for (let i = 0; i < 640 * 640; ++i) {
      input[i] = pixels[i * 4] / 255;
      input[640 * 640 + i] = pixels[i * 4 + 1] / 255;
      input[640 * 640 * 2 + i] = pixels[i * 4 + 2] / 255;
    }
  
    return [input, img_width, img_height];
  }
  
  async function run_model(input, model) {
    const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);
    const outputs = await model.run({ images: tensor });
    return outputs["output0"].data;
  }
  
  function process_output(output, img_width, img_height, bulu_classes) {
    const confThreshold = 0.3;
    const nmsThreshold = 0.2;
  
    let boxes = [];
    const jumlah_cls = bulu_classes.length;
    for (let index = 0; index < 8400; index++) {
      const [class_id, prob] = [...Array(jumlah_cls).keys()]
        .map(col => [col, output[8400 * (col + 4) + index]])
        .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);
      if (prob < confThreshold) continue;
  
      const label = bulu_classes[class_id];
      const xc = output[index];
      const yc = output[8400 + index];
      const w = output[2 * 8400 + index];
      const h = output[3 * 8400 + index];
      const x1 = (xc - w / 2) / 640 * img_width;
      const y1 = (yc - h / 2) / 640 * img_height;
      const x2 = (xc + w / 2) / 640 * img_width;
      const y2 = (yc + h / 2) / 640 * img_height;
      boxes.push([x1, y1, x2, y2, label, prob]);
    }
  
    const result = non_maximum_suppression(boxes, nmsThreshold);
    return result;
  }
  
  const non_maximum_suppression = (boxes, threshold) => {
    if (boxes.length === 0) return [];
  
    boxes.sort((a, b) => b[5] - a[5]);
    const pickedBoxes = [];
  
    while (boxes.length > 0) {
      const box = boxes.shift();
      pickedBoxes.push(box);
      boxes = boxes.filter(b => iou(box, b) < threshold);
    }
  
    return pickedBoxes;
  };
  
  const intersection = (box1, box2) => {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const x1 = Math.max(box1_x1, box2_x1);
    const y1 = Math.max(box1_y1, box2_y1);
    const x2 = Math.min(box1_x2, box2_x2);
    const y2 = Math.min(box1_y2, box2_y2);
    const interWidth = Math.max(0, x2 - x1);
    const interHeight = Math.max(0, y2 - y1);
    return interWidth * interHeight;
  };
  
  const union = (box1, box2) => {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
  };
  
  const iou = (box1, box2) => {
    const inter = intersection(box1, box2);
    const union0 = union(box1, box2);
    return inter / union0;
  };
  
  async function draw_boxes_on_image(buf, boxes) {
    const imgBitmap = await createImageBitmap(new Blob([buf]));
    const canvas = new OffscreenCanvas(imgBitmap.width, imgBitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgBitmap, 0, 0, imgBitmap.width, imgBitmap.height);
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 1;
    ctx.font = '10px Arial';
    ctx.fillStyle = 'green';
  
    boxes.forEach(([x1, y1, x2, y2, label, prob]) => {
      ctx.beginPath();
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.stroke();
      const text = `${label}`;
      ctx.fillText(text, x1, y1 > 20 ? y1 - 5 : y1 + 20);
    });
  
    return await canvas.convertToBlob({ type: 'image/jpeg' });
  }
  