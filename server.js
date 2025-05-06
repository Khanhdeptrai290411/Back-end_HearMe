const express = require('express');
const path = require('path');
const { PythonShell } = require('python-shell');
const app = express();
const port = 3000;

// Phục vụ các file tĩnh (HTML, CSS, video)
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '50mb' })); // Để nhận dữ liệu JSON lớn (như frame video)

// Danh sách chương và bài học
const roadmap = {
  "Chapter 1": [
    { name: "FLOWER", path: "/Color_video/509912909642189-FLOWER.mp4", embedding: "reference_embedding/509912909642189-FLOWER_embedding.npy" },
    { name: "CONGRATULATIONS", path: "/Color_video/164951232112037-CONGRATULATIONS.mp4", embedding: "reference_embedding/164951232112037-CONGRATULATIONS_embedding.npy" },
    { name: "WELCOME", path: "/Color_video/12579272885288595-WELCOME.mp4", embedding: "reference_embedding/12579272885288595-WELCOME_embedding.npy" },
    { name: "HELLO", path: "/Color_video/5339916560192981-HELLO.mp4", embedding: "reference_embedding/5339916560192981-HELLO.mp4" },
    { name: "DESIGN", path: "/Color_video/7119068123432775-DESIGN.mp4", embedding: "reference_embedding/7119068123432775-DESIGN_embedding.npy" }
  ],
  "Chapter 2": [
    { name: "BRIGHT", path: "/Color_video/18880274572777278-BRIGHT.mp4", embedding: "reference_embedding/18880274572777278-BRIGHT_embedding.npy" },
    { name: "GREET", path: "/Color_video/25196207384020064-GREET.mp4", embedding: "reference_embedding/25196207384020064-GREET_embedding.npy" },
    { name: "CHEER", path: "/Color_video/48093054817466707-CHEER.mp4", embedding: "reference_embedding/48093054817466707-CHEER_embedding.npy" }
  ]
};

// API để lấy danh sách chương và bài học
app.get('/api/roadmap', (req, res) => {
  res.json(roadmap);
});

// API để xử lý video từ webcam
app.post('/api/process-video', (req, res) => {
  const { frames, lessonPath } = req.body;

  // Tìm embedding tương ứng với bài học
  let referenceEmbeddingPath = '';
  Object.values(roadmap).forEach(chapter => {
    chapter.forEach(lesson => {
      if (lesson.path === lessonPath) {
        referenceEmbeddingPath = lesson.embedding;
      }
    });
  });

  if (!referenceEmbeddingPath) {
    return res.status(400).json({ error: 'Lesson not found' });
  }

  // Gọi script Python để xử lý frames
  const options = {
    mode: 'json',
    pythonOptions: ['-u'],
    scriptPath: path.join(__dirname, 'scripts'),
    args: [JSON.stringify(frames), referenceEmbeddingPath]
  };

  PythonShell.run('process_video.py', options, (err, results) => {
    if (err) {
      console.error('Python script error:', err);
      return res.status(500).json({ error: 'Error processing video' });
    }

    const result = results[0];
    res.json({
      similarity: result.similarity,
      status: result.similarity > 0.8 ? 'Match!' : 'Keep Practicing'
    });
  });
});

// Khởi động server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
}); 