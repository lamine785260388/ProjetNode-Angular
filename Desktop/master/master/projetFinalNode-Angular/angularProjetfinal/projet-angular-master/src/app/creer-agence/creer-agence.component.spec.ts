import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreerAgenceComponent } from './creer-agence.component';

describe('CreerAgenceComponent', () => {
  let component: CreerAgenceComponent;
  let fixture: ComponentFixture<CreerAgenceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CreerAgenceComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreerAgenceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
