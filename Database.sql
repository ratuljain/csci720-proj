
create database Authorship_Attribution;

use Authorship_Attribution;

CREATE TABLE author1
(
    authors nvarchar(100) NOT NULL,
    sentences text ,
	para_len int ,
	sent int ,
	sent_max_len int ,
	word int,
	unique_word int,
	stop_words int,
	comma int,
	special int,
	uppercase int
);

