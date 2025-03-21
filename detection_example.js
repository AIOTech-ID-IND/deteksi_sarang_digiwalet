//Kode di HTMLnya:
//<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
//<script src="{{ url_for('static', filename='js/detection_api.js') }}"></script>
//<script src="{{ url_for('static', filename='js/detection_example.js') }}"></script>

const { Tensor, InferenceSession } = ort;

// Path ke file model ONNX
const modelPath_bulu = 'models/model_bulu_walet.onnx';
const modelPath_bentuk = 'models/model_bentuk_walet.onnx';

// Model ONNX
let model_bulu, model_bentuk

// Memuat model ONNX sebagai variabel global
async function initializeModels() {
    try {
        model_bentuk = await InferenceSession.create(modelPath_bentuk);
        model_bulu = await InferenceSession.create(modelPath_bulu);

        console.log('Model ONNX berhasil dimuat.');
    } catch (err) {
        console.error('Gagal memuat model ONNX:', err);
    }
}

//  Inisilisasi model saat aplikasi dimulai
initializeModels();

// Fungsi untuk menghitung jumlah kelas dalam array hasil deteksi
function countClass(results, class_label) {
  console.log(results);
  const count = results.filter(result => result[4] === class_label).length;
  return count;
}

// Fungsi untuk melakukan encoding base64 dari buffer gambar
function base64EncodeImage(imageBuffer) {
  const blob = new Blob([imageBuffer]);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = reader.result.split(',')[1];
      resolve(base64String);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// Variabel file belum terdefinisi 
// Contohnya:
// var fileInput = document.getElementById('multiFiles');
// var file = fileInput.files[0];

// Deteksi Bulu
const classes_bulu = ["BR", "BR", "BR", "BS", "BS", "BB", "BB2"];
const results_bulu = await detect_objects_on_image(file, model_bulu, classes_bulu);
const base64_encoded_bulu = await base64EncodeImage(results_bulu[0]);

// Deteksi Bentuk
const classes_bentuk = ["Mangkok", "Oval", "Segitiga"];
const results_bentuk = await detect_objects_on_image(file, model_bentuk, classes_bentuk);
const base64_encoded_bentuk = await base64EncodeImage(results_bentuk[0]); 

// Hitung jumlah objek per kelas berdasarkan hasil deteksi
var jumlah_BRS = countClass(results_bulu[1], "BRS");
var jumlah_BR = countClass(results_bulu[1], "BR");
var jumlah_BST = countClass(results_bulu[1], "BST");
var jumlah_BS = countClass(results_bulu[1], "BS");
var jumlah_BBT = countClass(results_bulu[1], "BBT");
var jumlah_BB = countClass(results_bulu[1], "BB");
var jumlah_BB2 = countClass(results_bulu[1], "BB2");

var jumlah_BR_4 = jumlah_BRS + jumlah_BR + jumlah_BST;
var jumlah_BS_4 = jumlah_BS + jumlah_BBT;
var jumlah_BB_4 = jumlah_BB;
var jumlah_BB2_4 = jumlah_BB2;


var jumlah_Mangkok = countClass(results_bentuk[1], "Mangkok");
var jumlah_Oval = countClass(results_bentuk[1], "Oval");
var jumlah_Segitiga = countClass(results_bentuk[1], "Segitiga");
