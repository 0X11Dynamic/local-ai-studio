-- Chat file attachments: a JSON list of attachment records per message
-- (files live in <data dir>\attachments\<conversation id>\).
ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]';
