import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { UploadZone } from '../src/components/UploadZone';

describe('UploadZone Component', () => {
  it('renders empty dropzone instructions initially', () => {
    const onSelectFile = vi.fn();
    const onError = vi.fn();

    render(
      <UploadZone
        selectedFile={null}
        onSelectFile={onSelectFile}
        onError={onError}
      />
    );

    expect(screen.getByText(/Drop your bone radiograph here/i)).toBeInTheDocument();
  });

  it('displays file preview and name when a file is provided', async () => {
    const onSelectFile = vi.fn();
    const onError = vi.fn();
    const mockFile = new File(['dummy bytes'], 'test-femur.png', { type: 'image/png' });

    global.URL.createObjectURL = vi.fn(() => 'blob:mock-preview-url');
    global.URL.revokeObjectURL = vi.fn();

    render(
      <UploadZone
        selectedFile={mockFile}
        onSelectFile={onSelectFile}
        onError={onError}
      />
    );

    const fileNameElement = await screen.findByText('test-femur.png');
    expect(fileNameElement).toBeInTheDocument();
    expect(await screen.findByText(/READY FOR ANALYSIS/i)).toBeInTheDocument();
  });
});
