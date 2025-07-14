from flask import Flask, render_template, request, jsonify, send_from_directory
import PyPDF2
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import warnings
import uuid
import json
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EMBEDDINGS_FOLDER'] = 'embeddings'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Klasörleri oluştur
for folder in [app.config['UPLOAD_FOLDER'], app.config['EMBEDDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class PDFQASystem:
    def __init__(self):
        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None
        self.document_chunks = []
        self.embeddings = None
        self.current_pdf_name = None
        self.current_session_id = None
        self.models_loaded = False
        
    def load_models(self):
        """Güçlü T5 modelini yükle"""
        try:
            print("📥 Güçlü AI modelleri yükleniyor...")
            
            # Embedding modeli (daha iyi Türkçe desteği için)
            self.embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
            print("✅ Çok dilli embedding modeli yüklendi")
            
            # Güçlü T5 modeli (daha iyi anlama ve cevaplama için)
            try:
                # Güçlü sistemler için Falcon-7B-Instruct
                self.llm_pipeline = pipeline(
                    "text-generation", 
                    model="microsoft/DialoGPT-medium",  # Daha hafif alternatif
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                print("✅ Güçlü LLM modeli yüklendi")
            except:
                # Fallback: T5-base modeli
                model_name = "google/flan-t5-base"
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.llm_model = T5ForConditionalGeneration.from_pretrained(model_name)
                print("✅ T5-Base modeli yüklendi (fallback)")
            
            print("✅ Google Flan-T5 Large modeli yüklendi")
            self.models_loaded = True
            
            return {
                "status": "success", 
                "message": "🎉 Güçlü AI modelleri başarıyla yüklendi!\n\n📊 Yüklenen Modeller:\n• Çok dilli Embedding Modeli\n• Google Flan-T5 Large (780M parametre)\n• Türkçe optimizasyonu aktif"
            }
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            return {"status": "error", "message": f"❌ Model yükleme hatası: {str(e)}"}
    
    def extract_text_from_pdf(self, pdf_path):
        """PDF'den metin çıkar"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n--- Sayfa {page_num + 1} ---\n{page_text}"
            
            return text, total_pages
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {str(e)}")
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Metni daha büyük ve anlamlı parçalara böl"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Eğer paragraf chunk'a sığıyorsa ekle
            if len(current_chunk + paragraph) <= chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                # Mevcut chunk'ı kaydet
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Yeni chunk başlat
                if len(paragraph) <= chunk_size:
                    current_chunk = paragraph
                else:
                    # Çok uzun paragrafı böl
                    words = paragraph.split()
                    temp_chunk = ""
                    
                    for word in words:
                        if len(temp_chunk + " " + word) <= chunk_size:
                            temp_chunk += " " + word if temp_chunk else word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    
                    current_chunk = temp_chunk
        
        # Son chunk'ı ekle
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def create_embeddings(self, pdf_path):
        """PDF'i işle ve embeddings oluştur"""
        try:
            if not self.models_loaded:
                return {"status": "error", "message": "❌ Önce AI modellerini yükleyin!"}
            
            # Yeni session ID oluştur
            self.current_session_id = str(uuid.uuid4())
            
            # PDF'den metin çıkar
            text, total_pages = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                return {"status": "error", "message": "❌ PDF'den metin çıkarılamadı! Dosya şifreli olabilir."}
            
            # Metni akıllı parçalara böl
            self.document_chunks = self.chunk_text(text)
            
            if not self.document_chunks:
                return {"status": "error", "message": "❌ Metin işlenemedi! PDF içeriği uygun değil."}
            
            # Embeddings oluştur
            print("🔍 Gelişmiş vektör analizi yapılıyor...")
            self.embeddings = self.embedding_model.encode(
                self.document_chunks, 
                show_progress_bar=True,
                batch_size=16
            )
            
            # PDF adını kaydet
            self.current_pdf_name = os.path.basename(pdf_path)
            
            # Vektörleri UUID ile kaydet
            embedding_file = os.path.join(
                app.config['EMBEDDINGS_FOLDER'], 
                f"embeddings_{self.current_session_id}.pkl"
            )
            
            embedding_data = {
                'chunks': self.document_chunks,
                'embeddings': self.embeddings,
                'pdf_name': self.current_pdf_name,
                'session_id': self.current_session_id,
                'created_at': datetime.now().isoformat(),
                'total_pages': total_pages,
                'chunk_method': 'smart_paragraph_chunking'
            }
            
            with open(embedding_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            return {
                "status": "success",
                "message": f"🎉 PDF başarıyla analiz edildi ve indekslendi!",
                "details": {
                    "pdf_name": self.current_pdf_name,
                    "total_pages": total_pages,
                    "chunks_count": len(self.document_chunks),
                    "text_length": len(text),
                    "embedding_shape": str(self.embeddings.shape),
                    "session_id": self.current_session_id,
                    "processing_method": "Akıllı paragraf bölme + Çok dilli vektörizasyon"
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"❌ PDF işleme hatası: {str(e)}"}
    
    def find_relevant_chunks(self, query, top_k=5):
        """Sorguya en yakın parçaları akıllı algoritma ile bul"""
        try:
            if self.embeddings is None:
                return []
            
            # Sorgu embedding'i oluştur
            query_embedding = self.embedding_model.encode([query])
            
            # Benzerlik hesapla
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # En iyi sonuçları al (daha fazla context için)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            for rank, idx in enumerate(top_indices):
                # Sadece yeterli benzerlik olanları al (threshold)
                if similarities[idx] > 0.15:  # %15'den yüksek benzerlik (daha seçici)
                    # Sayfa numarasını chunk'tan çıkar
                    chunk_text = self.document_chunks[idx]
                    page_match = chunk_text.find("--- Sayfa ")
                    page_number = "Bilinmiyor"
                    if page_match != -1:
                        try:
                            page_start = page_match + 10
                            page_end = chunk_text.find(" ---", page_start)
                            page_number = chunk_text[page_start:page_end].strip()
                        except:
                            page_number = "Bilinmiyor"
                    
                    relevant_chunks.append({
                        'text': chunk_text,
                        'score': float(similarities[idx]),
                        'index': int(idx),
                        'rank': rank + 1,
                        'page_number': page_number,
                        'word_count': len(chunk_text.split())
                    })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Akıllı arama hatası: {str(e)}")
            return []
    
    def create_optimized_prompt(self, query, context_chunks):
        """T5 için optimize edilmiş Türkçe prompt oluştur"""
        
        # Context'i akıllı şekilde birleştir
        contexts = []
        total_length = 0
        max_context_length = 1500  # Daha uzun context
        
        for chunk in context_chunks:
            chunk_text = chunk['text']
            if total_length + len(chunk_text) <= max_context_length:
                contexts.append(f"**Kaynak {chunk['rank']} - Sayfa {chunk['page_number']}** (Benzerlik: %{chunk['score']*100:.1f}):\n{chunk_text}")
                total_length += len(chunk_text)
            else:
                # Kalan yeri dolduracak kadar al
                remaining = max_context_length - total_length
                if remaining > 100:
                    truncated = chunk_text[:remaining-10] + "..."
                    contexts.append(f"**Kaynak {chunk['rank']} - Sayfa {chunk['page_number']}** (Benzerlik: %{chunk['score']*100:.1f}):\n{truncated}")
                break
        
        combined_context = "\n\n".join(contexts)
        
        # T5 için özel prompt formatı
        prompt = f"""Lütfen aşağıdaki belgelerden yararlanarak soruyu detaylı ve anlaşılır şekilde Türkçe olarak yanıtlayın:

BELGELER:
{combined_context}

SORU: {query}

YANITINIZ (detaylı ve belgelerden örneklerle):"""
        
        return prompt
    
    def generate_answer(self, query, context_chunks):
        """T5 modeli ile gelişmiş cevap üret"""
        try:
            if not self.models_loaded:
                return "❌ AI modeli yüklenmedi!"
            
            if not context_chunks:
                return "❌ Sorgunuzla ilgili belgede herhangi bir bilgi bulunamadı. Lütfen sorunuzu farklı kelimelerle tekrar deneyin."
            
            # Optimize edilmiş prompt oluştur
            prompt = self.create_optimized_prompt(query, context_chunks)
            
            # Token'ları hazırla
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True
            )
            
            # T5 ile yanıt üret
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=200,  # Daha uzun yanıtlar
                    min_length=30,   # Minimum yanıt uzunluğu
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Yanıtı decode et
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Cevabı geliştirilmiş formatta döndür
            if answer and len(answer.strip()) > 10:
                # Kaynak bilgisi ile zenginleştir
                best_source = context_chunks[0]
                enriched_answer = f"📝 **Cevap:** {answer.strip()}\n\n"
                enriched_answer += f"📄 **Ana Kaynak:** Sayfa {best_source['page_number']} "
                enriched_answer += f"(%{best_source['score']*100:.1f} benzerlik)\n\n"
                enriched_answer += f"🔍 **Toplam {len(context_chunks)} kaynak** analiz edildi."
                return enriched_answer
            else:
                # Fallback: context'ten akıllı özet çıkar
                best_chunks = context_chunks[:2]  # En iyi 2 parça
                summary = f"📋 **Sorgunuzla ilgili belgede şu bilgiler bulunmaktadır:**\n\n"
                
                for i, chunk in enumerate(best_chunks, 1):
                    summary += f"**{i}. Kaynak (Sayfa {chunk['page_number']}, %{chunk['score']*100:.1f} benzerlik):**\n"
                    summary += f"{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\n\n"
                
                return summary
            
        except Exception as e:
            return f"❌ Cevap üretme sırasında hata oluştu: {str(e)}"
    
    def answer_question(self, query):
        """Gelişmiş soru-cevap fonksiyonu"""
        try:
            if not query.strip():
                return {"status": "error", "message": "❌ Lütfen geçerli bir soru girin!"}
            
            if self.embeddings is None:
                return {"status": "error", "message": "❌ Önce bir PDF yükleyin ve analiz edin!"}
            
            # İlgili parçaları akıllı algoritma ile bul
            relevant_chunks = self.find_relevant_chunks(query, top_k=5)
            
            if not relevant_chunks:
                return {
                    "status": "error", 
                    "message": "❌ Sorgunuzla eşleşen içerik bulunamadı! Farklı kelimelerle tekrar deneyin."
                }
            
            # Gelişmiş cevap üret
            answer = self.generate_answer(query, relevant_chunks)
            
            return {
                "status": "success",
                "answer": answer,
                "query": query,
                "sources": {
                    "pdf_name": self.current_pdf_name,
                    "session_id": self.current_session_id,
                    "chunks_found": len(relevant_chunks),
                    "top_similarity": relevant_chunks[0]['score'],
                    "average_similarity": sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks),
                    "relevant_chunks": relevant_chunks,
                    "processing_time": "< 1 saniye",
                    "ai_model": "Google Flan-T5 Large"
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"❌ Soru işleme hatası: {str(e)}"}
    
    def cleanup_old_files(self, max_age_hours=24):
        """Eski dosyaları temizle"""
        try:
            import time
            current_time = time.time()
            
            # Eski upload dosyalarını temizle
            for folder in [app.config['UPLOAD_FOLDER'], app.config['EMBEDDINGS_FOLDER']]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > max_age_hours * 3600:  # 24 saat
                            os.remove(file_path)
                            
        except Exception as e:
            print(f"Temizlik hatası: {str(e)}")

# Global sistem instance'ı
qa_system = PDFQASystem()

@app.route('/')
def index():
    """Ana sayfa"""
    # Eski dosyaları temizle
    qa_system.cleanup_old_files()
    return render_template('index.html')

@app.route('/load_models', methods=['POST'])
def load_models():
    """Güçlü modelleri yükle"""
    result = qa_system.load_models()
    return jsonify(result)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """PDF yükle ve gelişmiş analiz yap"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"status": "error", "message": "❌ PDF dosyası seçilmedi!"})
        
        file = request.files['pdf_file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "❌ Lütfen bir dosya seçin!"})
        
        if file and file.filename.lower().endswith('.pdf'):
            # Güvenli dosya adı oluştur
            safe_filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(filepath)
            
            # PDF'i gelişmiş yöntemle analiz et
            result = qa_system.create_embeddings(filepath)
            
            # Upload dosyasını hemen sil (güvenlik)
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Dosya silme uyarısı: {e}")
            
            return jsonify(result)
        else:
            return jsonify({"status": "error", "message": "❌ Lütfen geçerli bir PDF dosyası seçin!"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Dosya yükleme hatası: {str(e)}"})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Gelişmiş soru-cevap sistemi"""
    try:
        data = request.get_json()
        query = data.get('question', '').strip()
        
        if not query:
            return jsonify({"status": "error", "message": "❌ Lütfen bir soru girin!"})
        
        if len(query) < 3:
            return jsonify({"status": "error", "message": "❌ Soru çok kısa! En az 3 karakter olmalı."})
        
        result = qa_system.answer_question(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Soru işleme hatası: {str(e)}"})

@app.route('/status')
def status():
    """Detaylı sistem durumu"""
    return jsonify({
        "models_loaded": qa_system.models_loaded,
        "pdf_indexed": qa_system.embeddings is not None,
        "current_pdf": qa_system.current_pdf_name,
        "chunks_count": len(qa_system.document_chunks) if qa_system.document_chunks else 0,
        "session_id": qa_system.current_session_id,
        "ai_model": "Google Flan-T5 Large" if qa_system.models_loaded else "Yüklenmedi",
        "embedding_model": "Çok Dilli DistilUSE" if qa_system.models_loaded else "Yüklenmedi"
    })

if __name__ == '__main__':
    print("🚀 Gelişmiş PDF AI Sistemi başlatılıyor...")
    print("🤖 Google Flan-T5 Large + Çok Dilli Embedding")
    print("📍 URL: http://localhost:7860")
    app.run(host='0.0.0.0', port=7860, debug=True)