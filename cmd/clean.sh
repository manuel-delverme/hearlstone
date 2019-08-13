#!/bin/bash
echo "Compressing local logs"
tar -cf logs.tar.gz logs/
mkdir ~/trash
echo "Moving logs to trash"
mv logs.tar.gz ~/trash
echo "Cleaning up.."
rm -r logs/*
rm -r jobs/*
rm *.std.{out, err}
